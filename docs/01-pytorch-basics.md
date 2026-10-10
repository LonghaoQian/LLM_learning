# PyTorch and Neural-Network Basics

This guide covers the mechanics needed before implementing a Transformer:
tensors, batches, modules, gradients, losses, optimization, text tokenization,
and a simple language-model baseline.

## 1. Set up the environment

Create an isolated environment and install PyTorch using the command appropriate
for your operating system and accelerator from the official PyTorch installation
page. Add testing tools separately.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install pytest
```

Verify PyTorch and inspect accelerator support:

```python
import torch

print(torch.__version__)
print(torch.cuda.is_available())
```

Choose a device in this order when available: CUDA, Apple MPS, then CPU. Operands
and model parameters involved in one operation normally need to be on the same
device.

### Exercise

Create two tensors on the selected device, multiply them, and print their shape,
dtype, and device.

### Success check

You can explain why the environment and device selection matter, and `pytest -q`
can discover a small passing test.

## 2. Understand tensors

A tensor is a multidimensional array. Three properties matter constantly:

- **shape:** the size along each dimension;
- **dtype:** the representation, such as integer IDs or floating-point values;
- **device:** the CPU or accelerator holding the data.

Examples:

```text
scalar                 shape ()
feature vector         shape (C,)
matrix                 shape (T, C)
batch of token IDs     shape (B, T)
batch of embeddings    shape (B, T, C)
```

Practice tensor creation, indexing, slicing, reshaping, transposing, concatenation,
stacking, reductions, matrix multiplication, and broadcasting.

### Shape exercise

For `x` shaped `(B, T, C)` and a matrix shaped `(C, F)`, predict the shape of
`x @ weight` before running it. Try small concrete values and inspect individual
elements.

### Broadcasting exercise

Add a position matrix shaped `(T, C)` to a batch shaped `(B, T, C)`. Explain which
dimension is reused and why the operation is valid.

## 3. Learn automatic differentiation

PyTorch records operations involving tensors that require gradients. This creates
a computation graph. Calling `loss.backward()` traverses that graph backward and
accumulates derivatives in parameter `.grad` fields.

Important points:

- gradients accumulate instead of automatically replacing old values;
- `optimizer.zero_grad()` clears old gradients;
- `optimizer.step()` changes parameters using their gradients;
- `torch.no_grad()` disables gradient recording for evaluation;
- detaching a tensor disconnects it from earlier graph operations.

### `.grad`, gradient clearing, and graph lifetime

The computation graph and a parameter's `.grad` field are related, but they
store different things:

| Item | What it contains | Normal lifetime |
| --- | --- | --- |
| computation graph | operations connecting inputs to the result | normally freed after `backward()` |
| `.grad` | the accumulated derivative for a trainable leaf tensor | remains until explicitly cleared |

For a small scalar example:

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
x = torch.tensor(3.0)
y = w * x

print(w.grad)  # None: no derivative has been calculated yet
y.backward()
print(w.grad)  # tensor(3.): dy/dw = x
```

The `.grad` attribute is storage, not a command. Calling `backward()` calculates
derivatives and places them in the `.grad` fields of leaf tensors that require
gradients. Model parameters are leaf tensors. Intermediate results do not retain
their gradients by default; call `intermediate.retain_grad()` only when an
exercise or debugging task requires inspecting one.

PyTorch adds each new derivative to the existing `.grad` value. It does not
replace the old value:

```python
w = torch.tensor(2.0, requires_grad=True)

loss = w**2
loss.backward()
print(w.grad)  # tensor(4.)

loss = w**2  # a new forward pass creates a new computation graph
loss.backward()
print(w.grad)  # tensor(8.): old 4 plus new 4
```

Accumulation is useful when deliberately combining several mini-batches. In a
normal optimization loop, clear old gradients before calculating new ones:

```python
optimizer.zero_grad(set_to_none=True)
logits = model(inputs)            # forward pass: build graph and predict
loss = loss_fn(logits, targets)   # extend the graph to a scalar loss
loss.backward()                   # populate parameter .grad fields
optimizer.step()                  # update parameters using those gradients
```

There is no separate instruction that says "build the graph." PyTorch builds its
dynamic graph while it executes the forward tensor operations. Calling
`model(inputs)` invokes the model's `forward` method through `nn.Module.__call__`;
operations inside `forward` create the path from parameters and inputs to
`logits`. Calling `loss_fn(logits, targets)` then extends that path from `logits`
to `loss`. If `loss.backward()` is called, a parameter such as `w` receives the
total derivative of that loss with respect to itself:

```text
w.grad = d(loss) / d(w)
```

PyTorch follows every path from `loss` back to `w` and applies the chain rule.
More generally, calling `result.backward()` accumulates `d(result) / d(w)` in
`w.grad`. The result is normally a scalar; a non-scalar result must first be
reduced, for example with `.mean()`, or be given an explicit starting gradient.

The four jobs in one training iteration are therefore:

1. `zero_grad()` clears gradients left by earlier iterations.
2. `model(inputs)` and `loss_fn(...)` execute the forward computation, produce
   predictions and loss, and build the computation graph as they run.
3. `loss.backward()` follows that graph backward and accumulates
   `d(loss) / d(parameter)` in each parameter's `.grad` field.
4. `optimizer.step()` reads those `.grad` values and updates the parameters.

With raw trainable tensors and no optimizer, clear a gradient with
`parameter.grad = None`. Setting it to `None` lets PyTorch allocate a fresh
gradient on the next backward pass. `parameter.grad.zero_()` instead keeps the
existing gradient tensor and fills it with zeros.

By default, `backward()` frees saved intermediate graph data after using it. A
second backward pass through the exact same graph therefore fails:

```python
loss = w**2
loss.backward()
loss.backward()  # RuntimeError: the graph was already freed
```

The option is named `retain_graph=True`, not `keep_graph`. It preserves the graph
when the same forward result genuinely needs more than one backward pass:

```python
w.grad = None
loss = w**2
loss.backward(retain_graph=True)
loss.backward()
print(w.grad)  # tensor(8.): gradients still accumulated
```

`retain_graph=True` preserves operation history; it neither preserves nor clears
gradients. Normal training does not need it because every iteration performs a
new forward pass and creates a new graph. Keeping graphs unnecessarily also uses
extra memory. Do not confuse `retain_graph=True` with `retain_grad()`:

- `retain_graph=True` keeps a graph available for another backward traversal;
- `retain_grad()` asks PyTorch to populate `.grad` on one non-leaf intermediate.

A useful summary is: `backward()` produces gradients, `.grad` stores and
accumulates them, `zero_grad()` clears old gradients, and `retain_graph=True`
keeps the operation history for another backward pass.

### Exercise: linear regression twice

Generate points following approximately `y = ax + b`.

1. Represent `a` and `b` as raw trainable tensors.
2. Calculate mean squared error.
3. Call `backward()` and manually update the parameters.
4. Rebuild the experiment with `nn.Linear` and an optimizer.
5. Compare the learned values.

Intentionally omit gradient clearing for several iterations. Inspect the gradients
and explain why the behavior changes.

### Success check

You can explain what `backward()`, `zero_grad()`, and `step()` each do without
describing them all as “training the model.”

## 4. Build networks with `nn.Module`

An `nn.Module` organizes parameters and submodules. Its `forward` method defines
the computation from inputs to outputs.

Build a two-layer classifier:

```text
input -> Linear -> activation -> Linear -> logits
```

Learn these distinctions:

- a **parameter** is a learned value stored by the model;
- an **activation** is an intermediate result for a particular batch;
- a **hyperparameter** is selected by you, such as layer width;
- a **logit** is an unnormalized output score;
- a **probability** is a normalized value after softmax.

Inspect trainable values with `model.parameters()`. Count parameters by summing
`parameter.numel()`.

### Exercise

Train the classifier on a small synthetic problem. Print parameter shapes and
gradient norms. Make the network deliberately too small and then large enough;
compare the results.

## 5. Understand loss and optimization

Cross-entropy is the main loss for classification and next-token prediction. For
the correct class `y`, its basic form is:

```text
loss = -log P(y)
```

PyTorch cross-entropy accepts logits and internally uses a numerically stable
log-softmax. Do not apply softmax first unless a specific API requires
probabilities.

A standard optimization loop is:

```python
model.train()
for inputs, targets in loader:
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss = loss_fn(logits, targets)
    loss.backward()
    optimizer.step()
```

Learn to inspect initial loss, loss trends, gradient values, parameter changes,
and whether logits and targets have the shapes expected by the loss function.

## 6. Understand samples, sequences, and batches

A dataset provides individual examples. A data loader groups examples into
batches. Batching makes computation efficient and gives a less noisy estimate of
the average gradient.

For language modeling:

```text
input IDs     (B, T)
target IDs    (B, T)
logits        (B, T, V)
```

Logits contain one score for every vocabulary token, for every sequence position,
for every example.

### Batch-size intuition

- small batches use less memory but give noisier gradients;
- large batches use more memory and may require learning-rate changes;
- one optimizer step processes approximately `B * T` target tokens;
- gradient accumulation can approximate a larger effective batch.

### Exercise

Create a custom `Dataset` for a small collection of labeled vectors. Load it with
several batch sizes and inspect the final partial batch.

## 7. Turn text into token IDs

A model operates on numbers, so text must be tokenized. Start with a character
tokenizer because its behavior is visible.

1. Collect unique characters from a small text.
2. Assign each character an integer ID.
3. Implement `encode(text) -> list[int]`.
4. Implement `decode(ids) -> str`.
5. Test that supported text survives an encode/decode round trip.

Tokenizer tradeoffs:

| Tokenizer | Advantage | Limitation |
| --- | --- | --- |
| Character | Simple and transparent | Long sequences |
| Byte | Fixed vocabulary, no unknown text bytes | Often long sequences |
| Word | Intuitive units | Huge vocabulary and unknown words |
| BPE/subword | Practical balance | Requires a learned tokenizer |

Use characters or bytes first. Treat BPE as a later experiment.

## 8. Create next-token examples

Language-model training predicts the next token from earlier tokens. From one
window of `T + 1` tokens, create:

```text
tokens:  l a n g u a g e
input:   l a n g u a g
target:  a n g u a g e
```

The input and target both have length `T`; the target is shifted one position.
Split the corpus into training and validation regions before sampling windows to
avoid overlap and leakage across the split.

### Required tests

- `decode(encode(text)) == text` for supported text;
- token IDs remain within `[0, V)`;
- inputs and targets use an integer dtype;
- batch shapes equal `(B, T)`;
- every target is its source shifted by one token;
- training and validation windows come from separate regions.

### Success check

Print and decode one input/target pair. Verify every position manually.

## 9. Build a bigram language model

Before attention, build a model that predicts the next token using only the
current one. A learnable table shaped `(V, V)` is enough: selecting token `i`
returns `V` logits for the token that may follow it.

Implement:

- a forward pass from `(B, T)` IDs to `(B, T, V)` logits;
- optional cross-entropy loss when targets are provided;
- a small training loop;
- periodic training and validation loss;
- autoregressive generation by sampling the next-token distribution;
- checkpoint save and restore.

For roughly uniform initial predictions, expected cross-entropy is near `log(V)`.
This is a useful check against an obviously incorrect initial loss.

### Essential debugging exercise

Train repeatedly on one batch until the model nearly memorizes it. If it cannot
overfit one batch, do not start a longer run. Check the shifted data, logits, loss
shapes, gradients, registered parameters, and learning rate.

The bigram model estimates `P(next token | current token)`. It cannot distinguish
two contexts ending in the same current token. The Transformer will address this
by combining information across the context.

## Questions you should be able to answer

- What are shape, dtype, and device?
- What is broadcasting, and when can it hide a bug?
- What does `loss.backward()` calculate?
- Why are gradients cleared between steps?
- What is the difference between a parameter and an activation?
- Why does cross-entropy accept logits?
- Why are language-model targets shifted?
- What does batch size change mathematically and operationally?
- Why separate training and validation data?
- Why can a bigram model not use distant context?

## Completion checklist

- [ ] Environment and device selection work.
- [ ] Tensor and broadcasting exercises are complete.
- [ ] Raw-tensor and `nn.Module` regression both learn.
- [ ] A small classifier trains successfully.
- [ ] Tokenizer round-trip tests pass.
- [ ] Shifted batch tests pass.
- [ ] Bigram loss decreases and the model generates text.
- [ ] The bigram model overfits one batch.
- [ ] A saved model produces the same logits after restoration.

Next: [Self-attention and Transformer theory](02-self-attention.md).

