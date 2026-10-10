# Network Training and Online Datasets

This guide covers reliable model training, validation, checkpointing, online data
loading, tokenization decisions, generation, evaluation, and debugging. Begin it
after your mini-GPT passes shape and causal-leakage tests.

## 1. Use a safe training progression

Never begin with a long run. Increase scope only after each smaller check passes:

1. Inspect one raw text record.
2. Encode and decode it.
3. Inspect one shifted input/target example.
4. Run one forward pass.
5. Run one backward pass and inspect gradients.
6. Overfit one batch.
7. Overfit a tiny dataset.
8. Run a short training/validation experiment.
9. Start a longer experiment only after reviewing the evidence.

This progression localizes bugs. A falling loss in a large run does not prove that
the split, causal mask, evaluation logic, or checkpoints are correct.

## 2. Configure experiments explicitly

Record at least:

```text
random seed
dataset name, configuration, split, and revision
tokenizer type and vocabulary
context length
embedding size, heads, layers, and dropout
batch size and gradient accumulation
optimizer and learning rate
weight decay
training steps or token budget
warmup and decay schedule
evaluation and checkpoint intervals
device and precision
```

Store model configuration with every checkpoint. Print the parameter count and
configuration at the beginning of a run. Use separate output directories for
separate experiments, and do not silently overwrite or resume an unrelated run.

## 3. Write the training loop

The essential operations are:

```python
model.train()
optimizer.zero_grad(set_to_none=True)
logits, loss = model(inputs, targets)
loss.backward()
optimizer.step()
```

Real training also needs device transfer, logging, validation, non-finite loss
detection, optional gradient clipping and scheduling, checkpointing, and clean
interruption handling. Never put validation examples through optimizer updates.

### Tokens per step

Without gradient accumulation, one step predicts roughly:

```text
batch_size * context_length
```

tokens. With `A` accumulation steps, an optimizer update uses approximately:

```text
A * batch_size * context_length
```

Record tokens processed because step counts are difficult to compare when batch
or sequence length changes.

## 4. Use AdamW thoughtfully

AdamW maintains moving estimates related to the first and second moments of
gradients and applies decoupled weight decay. It is common for Transformers, but
it does not remove the need to understand the learning rate.

Learn these controls:

- **learning rate:** update scale;
- **weight decay:** regularization applied to selected weights;
- **warmup:** gradual early increase in learning rate;
- **decay schedule:** reduction later in training;
- **gradient clipping:** limits unusually large gradient norms.

Record the gradient norm before clipping while debugging. A threshold can prevent
one spike from destroying a run, but repeated clipping may indicate a deeper
problem.

Biases and normalization parameters are often excluded from weight decay. Treat
that as an explicit design choice rather than copied ritual.

## 5. Separate training and validation

Training loss uses data involved in parameter updates. Validation loss uses
held-out data and estimates generalization.

Validation should use:

```python
model.eval()
with torch.no_grad():
    ...
```

Return to `model.train()` afterward. Average over several validation batches
rather than reporting one random batch.

Typical interpretations:

- both losses high: underfitting, insufficient training, or a bug;
- both decrease: normal learning;
- training decreases while validation rises: likely overfitting or mismatch;
- validation below training: possible with dropout or measurement timing.

Perplexity is:

```text
perplexity = exp(average_cross_entropy)
```

Always state the tokenizer with perplexity. Values based on different token units
are not directly comparable.

## 6. Make checkpoints complete

A useful checkpoint contains:

- model state;
- optimizer state;
- scheduler state, if used;
- model and training configuration;
- tokenizer identity or serialized tokenizer;
- current step and tokens processed;
- best validation metric;
- random-number-generator state when exact continuation matters.

Keep the latest checkpoint separate from the best-validation checkpoint. Write
checkpoints atomically when possible so interruption does not leave a partial
file.

### Required checkpoint tests

- saved and reloaded models produce matching logits in evaluation mode;
- optimizer state is restored;
- training resumes from the expected step;
- configuration mismatch produces a clear warning or error;
- a prompt can be generated from only the documented artifacts.

## 7. Choose an online dataset

TinyStories is suitable for a first small language model because it uses simple
stories and restricted language. The associated research shows that relatively
small models can learn visible grammatical and narrative structure.

One accessible configuration is:

```python
from datasets import load_dataset

dataset = load_dataset("eminorhan/tinystories", "10M")
print(dataset)
print(dataset["train"][0]["text"])
```

Do not immediately train on all records. Begin with a small slice and validate the
whole pipeline.

### Before using any online dataset

1. Read its dataset card.
2. Check its license and intended use.
3. Inspect how it was collected or generated.
4. Inspect several examples and field names.
5. Check whether official splits exist.
6. Look for empty, duplicate, malformed, or extremely long records.
7. Record the exact name, configuration, split, and revision.
8. Decide whether to download, cache, or stream it.

Online data is external input. Validate fields and types, and fail with a useful
error instead of assuming every record contains valid text.

## 8. Downloading versus streaming

Downloading and preprocessing locally is often simplest for a first experiment.
It enables repeatable shuffling and inspection after the initial download.

Streaming avoids storing a complete large corpus:

```python
dataset = load_dataset(
    "eminorhan/tinystories",
    "10M",
    split="train",
    streaming=True,
)
```

Streaming changes dataset behavior:

- length may not be readily available;
- random access may be unavailable;
- shuffling usually uses a finite buffer;
- network interruption can affect training;
- reproducibility depends on revisions, seeds, buffers, and workers.

For this learning project, prefer a downloaded small subset until training and
checkpointing work. Use streaming later as a systems exercise.

## 9. Prepare stories for language modeling

A clear preprocessing pipeline is:

```text
raw record
  -> validate text field
  -> normalize only intentionally
  -> add an explicit story separator/end token
  -> tokenize
  -> concatenate or pack tokens
  -> create context windows
  -> form shifted inputs and targets
```

Decide how documents are separated. Without a separator, the model may treat the
end of one story and start of another as a normal continuation.

Avoid excessive cleaning. Case, punctuation, and newlines carry information.
Every normalization rule should have a reason and be recorded.

### Avoid leakage

Use official training and validation splits when trustworthy. If you create your
own, split at the document level before generating token windows. Splitting after
concatenation can place overlapping or adjacent content in both sets.

## 10. Decide on a tokenizer

Start with character or byte tokenization to validate the training system. Then
treat BPE as its own learning experiment.

To understand BPE:

1. represent training text as initial symbols;
2. count adjacent symbol-pair frequencies;
3. merge the most frequent pair;
4. repeat to a target vocabulary size;
5. save merge rules and vocabulary;
6. apply those exact rules to validation and prompt text.

Never train the tokenizer separately on validation data. Store it with the
checkpoint because token IDs only have meaning relative to their vocabulary.

Tokenizer changes alter sequence length, vocabulary projection size, and the unit
of loss. Do not change both tokenizer and model architecture in one experiment if
you want a clear comparison.

## 11. Plan the first TinyStories experiment

Begin with a configuration that fits comfortably:

```text
dataset subset        small, validated TinyStories sample
tokenizer             character or byte
context length        128
embedding size        128
attention heads       4
Transformer layers    4
dropout               0.1
optimizer             AdamW
```

Reduce embedding size, layers, context, or batch size for CPU training. Maintain
a separate tiny configuration for smoke tests.

Before a substantial run, report:

- parameter count;
- training and validation token counts;
- device and precision;
- batch size, context length, and effective tokens per update;
- maximum steps or token budget;
- output directory;
- a sample from the untrained model.

Do not start a long training job, download a large corpus, or use paid compute
without deliberate approval.

## 12. Evaluate more than loss

Track:

- training and validation cross-entropy;
- perplexity with tokenizer stated;
- tokens processed;
- tokens per second;
- parameter count;
- runtime;
- peak accelerator memory when available.

Also save generations from fixed prompts, sampling settings, seeds, and
checkpoints. Samples reveal repetition, prompt neglect, broken word structure, or
memorization that average loss may hide.

### Controlled experiments

Change one important variable at a time:

- bigram versus Transformer;
- one versus four Transformer layers;
- short versus longer context;
- character versus BPE tokenizer;
- with versus without dropout;
- different generation temperatures.

Record:

```text
Question:
Hypothesis:
Controlled variables:
Changed variable:
Result:
Interpretation:
Next experiment:
```

Distinguish measured results from guesses. Keep failed experiments; they are often
excellent interview discussion material.

## 13. Diagnose common failures

### Loss stays near `log(V)`

Check target shifting, token range, parameter registration, optimizer parameters,
gradients, logits/target shapes, and learning rate. Return to the one-batch test.

### Loss becomes NaN or infinite

Check learning rate, input validity, masking values, logits, normalization,
gradient norm, and mixed precision. Stop instead of saving a corrupted checkpoint.
Add finite-value assertions close to the first invalid operation.

### Training loss falls but validation worsens

Check overfitting, duplicate or leaked data, distribution differences, and the
evaluation procedure. Consider more data or regularization only after verifying
correctness.

### Generation ignores the prompt

Confirm that prompt IDs are passed in, the latest context is retained, final
position logits are sampled, evaluation mode is active, and tokenizer/checkpoint
pairs match.

### Training is unexpectedly slow

Measure data loading, forward/backward time, evaluation time, tokens per second,
and device utilization. Change configuration only after locating the bottleneck.

### Out-of-memory error

Reduce batch size first, then context length or model dimensions. Standard
attention storage grows approximately with `T^2`. Gradient accumulation can
restore effective batch size but cannot reduce memory for one microbatch.

## 14. Experiment report template

```markdown
# Experiment name

## Question and hypothesis

## Environment
- date:
- hardware:
- PyTorch version:
- seed:

## Data
- dataset/configuration/revision:
- splits and sample counts:
- preprocessing:
- tokenizer:

## Model and training
- parameter count:
- architecture:
- optimizer and schedule:
- batch/context size:
- steps and tokens processed:

## Results
- best validation loss:
- perplexity:
- runtime and throughput:
- fixed-prompt samples:

## Interpretation
- what worked:
- what failed:
- limitations:
- next experiment:
```

## Questions you should be able to answer

- Why must validation avoid gradient updates?
- What does AdamW store beyond model parameters?
- What does gradient clipping solve, and what can it hide?
- What must a resumable checkpoint contain?
- Why can online dataset streaming reduce reproducibility?
- How can preprocessing cause train/validation leakage?
- Why are perplexities from different tokenizers difficult to compare?
- How would you diagnose a NaN loss?
- How do batch size and context length affect memory and tokens per update?
- How would you decide whether a model is data-limited or capacity-limited?

## Completion checklist

- [ ] One batch and a tiny dataset can be overfit.
- [ ] Training and validation modes are used correctly.
- [ ] Non-finite loss is detected.
- [ ] Checkpoints restore all required state.
- [ ] Dataset card, schema, and license were reviewed.
- [ ] Documents are split before window generation.
- [ ] Tokenizer metadata is stored with the model.
- [ ] A small online dataset subset trains successfully.
- [ ] Metrics, hardware, configuration, and samples are recorded.
- [ ] At least one controlled comparison is documented.

## Primary references

- [PyTorch: Learn the Basics](https://docs.pytorch.org/tutorials/beginner/basics/)
- [PyTorch: Deep Learning for NLP](https://docs.pytorch.org/tutorials/beginner/nlp/)
- [Hugging Face Datasets: Loading data](https://huggingface.co/docs/datasets/en/loading)
- [TinyStories paper](https://arxiv.org/abs/2305.07759)
- [TinyStories 10M dataset page](https://huggingface.co/datasets/eminorhan/tinystories)

