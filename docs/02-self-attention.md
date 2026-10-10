# Self-Attention and Transformer Theory

This guide explains the core theory behind a decoder-only Transformer. Work
through it after you can create correct next-token batches and train the bigram
baseline from the PyTorch basics guide.

The goal is to understand every shape and information path, then implement the
operations with ordinary PyTorch tensors and small layers.

## 1. Why attention is needed

A bigram model sees only the current token when predicting the next one. A useful
language model must combine information from earlier positions:

```text
The animal did not cross the road because it ___
```

The meaning of `it` and a plausible continuation may depend on distant words.
Self-attention creates a new representation for each position by taking a
content-dependent weighted mixture of other allowed positions.

For a causal language model, “allowed” means the current and earlier positions,
never future positions.

## 2. Embeddings and position

Token IDs shaped `(B, T)` enter an embedding table and become vectors:

```text
token IDs             (B, T)
token embeddings      (B, T, C)
position embeddings   (T, C)
combined input        (B, T, C)
```

Token embeddings say what token is present. Positional information distinguishes
the same token at different sequence positions. Without position information,
attention alone does not naturally know token order.

### Exercise

Create token and learned position embeddings. Add them using broadcasting. Explain
why the `(T, C)` position table is reused for each of the `B` examples.

## 3. Queries, keys, and values

Given input representations `X` shaped `(B, T, C)`, learned projections produce:

```text
Q = X W_Q
K = X W_K
V = X W_V
```

For a single head of size `D`, Q, K, and V each have shape `(B, T, D)`.

One useful intuition:

- a **query** describes what the current position is looking for;
- a **key** describes what each candidate position offers;
- a **value** contains the information copied when that candidate is selected.

Queries and keys determine weights. Values determine the information being mixed.
All three are learned transformations of the same input in self-attention.

## 4. Calculate compatibility scores

For every query position, compare it with every key position using a dot product:

```text
scores = Q @ K.transpose(-2, -1)
```

Shape derivation:

```text
(B, T, D) @ (B, D, T) -> (B, T, T)
```

The last two dimensions form a table. Row `i`, column `j` describes how strongly
position `i` attends to position `j` before masking and normalization.

### Why scale the scores?

Scaled dot-product attention divides by `sqrt(D)`:

```text
scores = scores / sqrt(D)
```

If query and key components have roughly unit variance, a sum of `D` products
tends to grow in variance with `D`. Large scores can saturate softmax and weaken
gradients. Scaling keeps scores in a more useful range.

## 5. Apply the causal mask

During next-token training, the full target sequence is present in memory. Without
a mask, position `i` could inspect positions after `i` and read the answer.

A causal mask allows the diagonal and lower triangle:

```text
allowed for T = 4

1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Replace forbidden score entries with negative infinity, or an appropriate very
negative value, **before** softmax. After softmax, forbidden positions receive
probability zero while each allowed row sums to one.

Masking after softmax is wrong because forbidden entries would already have
consumed probability mass.

## 6. Normalize and mix values

Complete one attention head:

```text
scores  = (Q K^T) / sqrt(D)
scores  = apply_causal_mask(scores)
weights = softmax(scores, last_dimension)
output  = weights V
```

Shapes:

```text
weights                 (B, T, T)
V                       (B, T, D)
output                   (B, T, D)
```

Each output position is a weighted sum of value vectors from allowed positions.
Attention weights depend on the data: different sequences and positions can
select different mixtures.

### Required tests for one head

- output shape equals `(B, T, D)`;
- every attention row sums to approximately one;
- weights above the causal diagonal are zero;
- outputs and gradients are finite;
- changing future input positions cannot change an earlier output in evaluation
  mode.

The last test is especially valuable. Run the head twice, modify only positions
after time `t`, and verify that outputs through `t` remain unchanged.

## 7. Multi-head attention

One attention pattern may specialize in one kind of relation. Multi-head attention
uses several learned projections in parallel.

With `H` heads and `D = C / H`:

```text
input                    (B, T, C)
Q, K, V reshaped         (B, H, T, D)
scores                   (B, H, T, T)
per-head outputs         (B, H, T, D)
joined output            (B, T, C)
output projection        (B, T, C)
```

The implementation may use separate modules per head for clarity or one combined
projection followed by reshaping. Begin with whichever makes the shapes easiest
to verify. Later, compare the implementations.

The output projection mixes information across concatenated heads and returns the
channel dimension `C`.

### Constraints and tests

- `C` must normally be divisible by `H`;
- the causal mask must broadcast over batch and head dimensions;
- reshaping must preserve intended position/head ordering;
- joined output must have shape `(B, T, C)`;
- invalid `C / H` configurations should fail clearly;
- the complete layer must pass the future-token leakage test;
- examples within a batch must not affect each other.

## 8. Computational cost

The score tensor has shape `(B, H, T, T)`. Standard full attention therefore has
quadratic time and memory growth in sequence length `T` for its attention matrix.

Doubling `T` approximately quadruples the number of pairwise attention scores.
This is a central reason long-context training is expensive.

The projections also cost computation, approximately proportional to `T * C^2`.
Which term dominates depends on model width and context length.

## 9. From attention to a Transformer block

Attention is one part of a Transformer block. A common pre-normalized decoder
block is:

```text
x = x + causal_self_attention(layer_norm_1(x))
x = x + feed_forward(layer_norm_2(x))
```

### Residual connections

A residual connection adds a sublayer's transformation back to its input. It
preserves a direct information and gradient path, making deep networks easier to
optimize. Input and output shapes must match.

### Layer normalization

Layer normalization standardizes features within each token representation and
then applies learned scale and bias. In a pre-norm Transformer, normalization
happens before each main sublayer.

### Feed-forward network

The position-wise feed-forward network independently transforms each position:

```text
(B, T, C) -> Linear(C, 4C) -> activation -> Linear(4C, C)
```

It mixes feature channels, while attention mixes information across sequence
positions.

### Dropout

Dropout randomly removes some activations during training as regularization. It is
disabled during evaluation, which is why deterministic leakage tests should use
`model.eval()`.

## 10. Assemble a decoder-only model

The full data path is:

```text
token IDs
  -> token embeddings + position embeddings
  -> repeated Transformer blocks
  -> final layer normalization
  -> vocabulary projection
  -> next-token logits
```

Recommended first learning configuration:

```text
context length       128
embedding size       128
attention heads      4
Transformer layers   4
dropout              0.1
```

Use a smaller test configuration, such as `C = 32` and two layers, for fast CPU
tests.

The vocabulary projection converts `(B, T, C)` to `(B, T, V)`. Cross-entropy then
compares all `B * T` predictions with shifted targets.

## 11. Generation

Autoregressive generation repeats:

1. keep no more than the most recent `context_length` prompt tokens;
2. run the model;
3. take logits from the final position;
4. convert them to a sampling distribution;
5. choose one next token;
6. append it and repeat.

Training predicts all positions in parallel because the known sequence and causal
mask are available. Generation is sequential because the next input token does
not exist until it has been selected.

Sampling methods include greedy decoding, temperature scaling, top-k sampling,
and optionally top-p sampling. They change output behavior, not learned weights.

## Model correctness checklist

- [ ] Token IDs have shape `(B, T)`.
- [ ] Embeddings have shape `(B, T, C)`.
- [ ] Q, K, and V have expected head dimensions.
- [ ] Scores have shape `(B, H, T, T)`.
- [ ] Forbidden attention weights are zero.
- [ ] Future-token leakage tests pass.
- [ ] Joined attention output has shape `(B, T, C)`.
- [ ] Residual operands have identical shapes.
- [ ] Vocabulary logits have shape `(B, T, V)`.
- [ ] Loss and gradients are finite.
- [ ] The complete model overfits one small batch.

## Questions you should be able to answer

- What does a query-key dot product mean?
- Why are values separate from keys?
- Why divide scores by `sqrt(D)`?
- Why must the causal mask be applied before softmax?
- What does one row of the attention matrix represent?
- What benefit do multiple heads provide?
- Why does attention scale quadratically with context length?
- What different roles do attention and the feed-forward network play?
- Why are residual connections useful?
- Why is generation sequential even though training is parallel across positions?
- How would you prove with a test that future information does not leak?

Next: [Network training and online datasets](03-training-and-online-data.md).

