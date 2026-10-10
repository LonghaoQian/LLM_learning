# Detailed Step-by-Step Build Roadmap

This document turns the learning material into a concrete implementation plan.
Follow it in order. Each milestone should leave the repository runnable and easier
to understand than before.

You should write the implementation. Use the coding assistant as a tutor and
reviewer unless you explicitly request implementation help.

Related theory:

- [PyTorch and neural-network basics](01-pytorch-basics.md)
- [Self-attention and Transformer theory](02-self-attention.md)
- [Network training and online datasets](03-training-and-online-data.md)

## Working method

For each step:

1. Read the linked concepts.
2. Write down the expected inputs, outputs, and tensor shapes.
3. Implement the smallest version that can work.
4. Add the listed tests.
5. run the tests and the manual observation.
6. Explain the result in your own words.
7. Ask for review if the step is marked as a review gate.
8. Commit the working milestone before adding the next concept.

Do not begin a long training run to compensate for a failing unit test. Fix the
smallest failing layer first.

## Target repository layout

Create files only when their milestone calls for them:

```text
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── configs/
│   ├── smoke.toml
│   └── tinystories-small.toml
├── data/                       # downloaded/generated data; ignored by Git
├── docs/
│   ├── learning-guide.md
│   ├── 01-pytorch-basics.md
│   ├── 02-self-attention.md
│   ├── 03-training-and-online-data.md
│   ├── 04-build-roadmap.md
│   └── experiments/
├── scripts/
│   └── inspect_dataset.py
├── src/
│   └── minigpt/
│       ├── __init__.py
│       ├── config.py
│       ├── device.py
│       ├── tokenizer.py
│       ├── data.py
│       ├── bigram.py
│       ├── attention.py
│       ├── model.py
│       ├── train.py
│       ├── evaluate.py
│       ├── generate.py
│       └── checkpoint.py
└── tests/
    ├── test_tokenizer.py
    ├── test_data.py
    ├── test_bigram.py
    ├── test_attention.py
    ├── test_model.py
    └── test_checkpoint.py
```

This is the final direction, not a request to create every empty file now.

## Milestone 0: Record the project contract

### Goal

Make the learning objective and boundaries visible to anyone opening the repo.

### Create or verify

- `README.md`
- `.gitignore`
- existing files under `docs/`

### Tasks

1. In the README, describe the project in two or three sentences.
2. State that it is a decoder-only language model built from scratch in PyTorch.
3. Link to `docs/learning-guide.md` and this roadmap.
4. Add a progress checklist linking to each milestone.
5. Ignore `.venv/`, Python caches, downloaded data, experiment outputs,
   checkpoints, and local secrets.
6. Decide on a license before sharing the repository publicly.

### Verification

- A new reader can identify the project's purpose and current milestone.
- Generated data and checkpoints do not appear as Git changes.
- Documentation links resolve inside the repository.

### Done when

The README answers: What is this? What will I learn? Where do I begin? What have I
completed?

## Milestone 1: Create the Python environment

### Goal

Run a reproducible Python package and a first test.

### Concepts

Virtual environments, dependencies, importable packages, test discovery, and
device selection.

### Create

- `pyproject.toml`
- `src/minigpt/__init__.py`
- `src/minigpt/device.py`
- one minimal test under `tests/`

### Tasks

1. Create and activate `.venv`.
2. Install the correct PyTorch build for the machine.
3. Configure the project as an installable `src`-layout package.
4. Add `pytest` as a development dependency.
5. Implement one function that selects CUDA, then MPS, then CPU.
6. Make a test assert that the function returns a valid PyTorch device.
7. Print Python, PyTorch, and device information from a small manual check.

### Commands you should eventually be able to run

```bash
python -m pip install -e .
pytest -q
python -c "import minigpt; print(minigpt.__name__)"
```

### Done when

- the package imports without editing `PYTHONPATH`;
- the test passes on CPU even if no accelerator exists;
- you can explain editable installation and the `src` layout.

### Review gate 1

Ask for review of dependency choices, package layout, and device-selection logic.

## Milestone 2: Complete tensor and autograd exercises

### Goal

Understand the mechanics that every later model component uses.

### Where to work

Use a temporary notebook, a scratch script, or `docs/notes/`. Do not add these
exercises to the production model modules.

### Tasks

1. Create tensors with different shapes and dtypes.
2. Practice indexing and slicing.
3. Convert `(B, T, C)` to `(B, C, T)` and back.
4. Multiply `(B, T, C)` by `(C, F)` and verify `(B, T, F)`.
5. Add `(T, C)` positions to `(B, T, C)` through broadcasting.
6. Write scalar linear regression with raw trainable tensors.
7. Inspect gradients before and after `backward()`.
8. Observe gradient accumulation when gradients are not cleared.
9. Repeat regression with `nn.Linear` and an optimizer.
10. Train a two-layer classifier on synthetic data.

### Questions to answer in your notes

- Which operations change shape without changing element order?
- What does a transpose change?
- When does broadcasting create a valid but unintended result?
- Why is a token ID normally an integer but an embedding floating point?
- What is stored in `.grad`?
- Which objects does an optimizer update?

### Done when

Both regression implementations learn approximately the generating relationship,
and you can write the standard optimization loop without reference material.

## Milestone 3: Build a character tokenizer

### Goal

Convert between human-readable text and model-readable IDs.

### Create

- `src/minigpt/tokenizer.py`
- `tests/test_tokenizer.py`
- a tiny text fixture under `tests/fixtures/` if needed

### Define the interface before implementation

Your tokenizer should expose clear equivalents of:

```text
build vocabulary from training text
encode text -> token IDs
decode token IDs -> text
vocabulary size
serialize/restore tokenizer metadata
```

Do not copy an external tokenizer for this milestone.

### Decisions to make

- How are vocabulary IDs ordered deterministically?
- What happens for a character absent from the vocabulary?
- Will there be an explicit end-of-document token?
- What information must be saved to reconstruct the same ID mapping?

### Required tests

- supported text round-trips through encode and decode;
- vocabulary IDs are unique and contiguous;
- vocabulary ordering is deterministic;
- invalid IDs fail clearly;
- unknown characters follow the documented policy;
- serialization and restoration preserve mappings.

### Manual observation

Print a short string, its IDs, and the decoded string. Explain each number.

### Done when

No model code needs to know how characters map to IDs.

### Review gate 2

Ask for an API and test review before building the dataset around this tokenizer.

## Milestone 4: Build next-token data windows

### Goal

Produce correct input/target batches for autoregressive training.

### Create

- `src/minigpt/data.py`
- `tests/test_data.py`

### Tasks

1. Encode a small corpus into one token stream.
2. Split training and validation regions before sampling windows.
3. Given a context length `T`, take windows of `T + 1` IDs.
4. Use the first `T` as input and last `T` as target.
5. Return integer tensors shaped `(B, T)`.
6. Support deterministic sampling when a generator or seed is provided.
7. Decide whether the last incomplete region is discarded or handled specially.

### Required tests

- each target is its input shifted by one position;
- both tensors have shape `(B, T)` and integer dtype;
- sampled positions never exceed the token stream;
- short corpora produce a clear error;
- fixed randomness reproduces a batch;
- training sampling cannot access validation positions.

### Manual observation

Decode every input and target in a tiny batch and align them side by side. Do not
continue until the one-token shift is visually obvious.

### Done when

The batch generator can feed any model that accepts `(B, T)` token IDs.

## Milestone 5: Implement the bigram baseline

### Goal

Train and sample from the simplest complete language model.

### Create

- `src/minigpt/bigram.py`
- `tests/test_bigram.py`
- a small executable training entry point or temporary training script

### Model contract

```text
input IDs              (B, T)
embedding lookup       (B, T, V)
logits                 (B, T, V)
optional targets       (B, T)
optional loss          scalar
```

The lookup table directly represents logits for the next token.

### Tasks

1. Implement the model forward pass.
2. Flatten logits and targets correctly for cross-entropy.
3. Train for a few steps on the tiny text.
4. Record initial and final losses.
5. Implement basic autoregressive sampling.
6. Try a fixed random seed for reproducible generation.
7. Save and restore model weights.

### Required tests

- logits have shape `(B, T, V)`;
- loss is scalar and finite;
- backward creates finite gradients;
- one optimizer step changes at least one parameter;
- restored weights reproduce logits in evaluation mode;
- generated token IDs stay within vocabulary range.

### Experiments

1. Compare initial loss with `log(V)`.
2. Overfit one batch.
3. Train on the small corpus.
4. Compare untrained and trained samples.

### Done when

Loss decreases, the one-batch test nearly memorizes its examples, and trained
samples show more local text structure than untrained samples.

### Review gate 3

Ask for review of the loss reshaping, gradient flow, and generation loop. Be ready
to explain why the model is limited to one-token context.

## Milestone 6: Separate reusable evaluation and generation

### Goal

Avoid mixing inference behavior into the training logic.

### Create

- `src/minigpt/evaluate.py`
- `src/minigpt/generate.py`

### Tasks

1. Move average-loss estimation into an evaluation function.
2. Make it use evaluation mode and no gradient tracking.
3. Restore the model's prior training/evaluation mode afterward.
4. Move sampling into a reusable function.
5. Support greedy selection and temperature.
6. Add top-k only after basic sampling is correct.
7. Validate invalid temperatures, prompts, and generation lengths.

### Required checks

- evaluation does not populate new gradients;
- repeated evaluation with dropout disabled is deterministic;
- generation never passes more than the supported context into the model;
- temperature changes probability sharpness as expected;
- top-k never samples outside the retained candidates.

### Done when

The bigram training script calls shared evaluation and generation functions rather
than containing duplicate logic.

## Milestone 7: Implement one causal attention head

### Goal

Build and prove the central attention operation before creating a Transformer.

### Read first

Sections 2–6 of `02-self-attention.md`.

### Create

- `src/minigpt/attention.py`
- `tests/test_attention.py`

### Write the shape plan first

```text
input X                (B, T, C)
Q, K, V                (B, T, D)
scores                 (B, T, T)
masked weights         (B, T, T)
output                 (B, T, D)
```

### Implementation order

1. Add learned Q, K, and V projections.
2. Calculate query-key dot products.
3. Divide by `sqrt(D)`.
4. Create or register a lower-triangular causal mask.
5. Mask forbidden scores before softmax.
6. Normalize along the key-position dimension.
7. Multiply attention weights by V.
8. Add dropout only after deterministic tests pass.

### Required tests

- output shape is correct;
- each weight row sums approximately to one;
- weights above the diagonal are zero;
- changing a future token cannot change an earlier output;
- changing one batch item cannot affect another;
- loss through the head creates gradients for Q, K, and V projections;
- sequence lengths up to the configured context work;
- oversized sequences fail clearly.

### Manual observation

For `T = 4`, print the mask and attention weights. Explain every row.

### Done when

The future-token leakage test passes, not merely the mask-shape test.

### Review gate 4

Ask for a detailed correctness review. Include your shape derivation and leakage
test, not just the module implementation.

## Milestone 8: Extend to multi-head attention

### Goal

Run multiple attention heads and return to the model channel dimension.

### Shape plan

```text
input                   (B, T, C)
split Q, K, V           (B, H, T, D)
scores                  (B, H, T, T)
head outputs            (B, H, T, D)
joined heads            (B, T, C)
projected output        (B, T, C)
```

### Tasks

1. Assert that `C` is divisible by `H`.
2. Choose separate-head modules or combined projections.
3. Make head/position transposes explicit and documented.
4. Concatenate heads in the correct order.
5. Add the output projection.
6. Add attention and residual dropout as separate concepts if desired.

### Required tests

- every intermediate and final shape matches the plan;
- `C % H != 0` fails with an informative message;
- the causal leakage test still passes;
- gradients reach all projections;
- dropout is active only during training;
- a one-head configuration behaves as a valid edge case.

### Done when

The layer accepts and returns `(B, T, C)` while preserving causality.

## Milestone 9: Build the feed-forward network and Transformer block

### Goal

Combine attention, feature transformation, normalization, and residual paths.

### Create or extend

- `src/minigpt/model.py`
- `tests/test_model.py`

### Implementation order

1. Implement a position-wise feed-forward network.
2. Verify `(B, T, C) -> (B, T, C)`.
3. Add two independent layer-normalization modules.
4. Build a pre-normalized attention residual path.
5. Build a pre-normalized feed-forward residual path.
6. Confirm residual operands have identical shapes.
7. Test one block before stacking blocks.

### Required tests

- feed-forward output shape equals its input shape;
- each sequence position is transformed independently by the feed-forward module;
- block output shape equals input shape;
- block gradients are finite;
- the full block passes the future-token leakage test;
- zeroing appropriate sublayer outputs demonstrates the residual path.

### Done when

One tested Transformer block maps `(B, T, C)` to `(B, T, C)` without future
information leakage.

## Milestone 10: Assemble the mini-GPT

### Goal

Construct the complete decoder-only language model.

### Components in order

1. Token embedding table.
2. Learned position embedding table.
3. Sum of token and position representations.
4. A stack of Transformer blocks.
5. Final layer normalization.
6. Vocabulary projection.
7. Optional cross-entropy loss.

### Configuration to start

Use two configurations:

```text
smoke/test:
  context length       16
  embedding size       32
  heads                4
  layers               2
  dropout              0.0

learning run:
  context length       128
  embedding size       128
  heads                4
  layers               4
  dropout              0.1
```

### Required tests

- token and position embeddings combine to `(B, T, C)`;
- logits have shape `(B, T, V)`;
- loss is scalar and finite;
- all expected parameters receive gradients;
- sequences longer than context length fail clearly;
- full-model future-token leakage test passes;
- parameter count is stable for a fixed configuration.

### Experiments

1. Count parameters by component.
2. Overfit one batch.
3. Overfit the tiny corpus.
4. Compare validation loss against the bigram model.
5. Generate from the same fixed prompt with both models.

### Done when

The model overfits one batch and beats the bigram baseline on data with useful
longer-range patterns.

### Review gate 5

Request an architecture review. Explain each residual path, normalization layer,
and major parameter group.

## Milestone 11: Add serializable configuration

### Goal

Make experiments reproducible without hard-coded hyperparameters.

### Create

- `src/minigpt/config.py`
- `configs/smoke.toml`
- `configs/tinystories-small.toml` later, after inspecting data

### Tasks

1. Define model and training configuration separately.
2. Validate divisibility, positive sizes, dropout range, and context length.
3. Load configuration from a human-readable file.
4. Print the resolved configuration at run start.
5. Save the resolved configuration with outputs and checkpoints.
6. Make command-line overrides explicit and record their final values.

### Required tests

- valid configuration round-trips through serialization;
- invalid dimensions fail before model construction;
- unknown fields do not silently disappear;
- defaults are visible and documented.

### Done when

The same configuration can reconstruct the same model architecture without
reading training code.

## Milestone 12: Build the robust trainer

### Goal

Train, validate, log, interrupt, and resume without losing experiment identity.

### Create or expand

- `src/minigpt/train.py`
- `src/minigpt/checkpoint.py`
- `tests/test_checkpoint.py`

### Implementation order

1. Move the proven small training loop into the trainer.
2. Add seeded randomness.
3. Add periodic averaged training and validation estimates.
4. Log learning rate, gradient norm, and tokens processed.
5. Stop with a diagnostic on non-finite loss.
6. Add optional gradient clipping.
7. Add a scheduler only after constant-rate training works.
8. Save latest and best-validation checkpoints separately.
9. Restore model, optimizer, scheduler, configuration, step, and tokenizer.
10. Refuse to overwrite an unrelated experiment silently.

### Smoke test

Use the tiny configuration and tiny corpus. The test should finish quickly on CPU
and produce a checkpoint that can generate text.

### Required tests

- checkpoint round-trip preserves logits;
- optimizer state restores;
- resume continues from the recorded step;
- evaluation does not change parameters;
- best and latest checkpoint rules differ correctly;
- non-finite loss stops rather than continuing;
- fixed seeds reproduce the parts documented as deterministic.

### Done when

You can interrupt a short run, resume it, and generate from the restored model.

### Review gate 6

Ask for review of evaluation isolation, checkpoint completeness, and resume
behavior before downloading an online dataset.

## Milestone 13: Inspect TinyStories without training

### Goal

Understand external data before connecting it to the model.

### Create

- `scripts/inspect_dataset.py`
- an inspection note under `docs/experiments/`

### Tasks

1. Read the dataset card and license.
2. Load only a small slice of the selected configuration.
3. Print schema and split information.
4. Inspect several stories manually.
5. Measure text length statistics on the slice.
6. Count empty or malformed text records.
7. Look for duplicate samples in the inspected subset.
8. Decide how an end-of-story separator will be represented.
9. Record dataset identifier, configuration, split, and revision.

### Questions to answer

- Is the dataset synthetic, human-authored, or mixed?
- Which field contains training text?
- Are train and validation splits supplied?
- Does the license permit the intended use?
- How much data can the machine store and process comfortably?
- What biases or limitations follow from simplified children's stories?

### Done when

The inspection note justifies the chosen subset and preprocessing policy. No long
training has started.

## Milestone 14: Build the online-data adapter

### Goal

Feed validated TinyStories text through the existing tokenizer and batch pipeline.

### Extend

- `src/minigpt/data.py`
- data tests using local fake records, not network calls

### Tasks

1. Accept a record iterator rather than coupling model code to one provider.
2. Validate that each record has a non-empty string field.
3. Add a documented separator between stories.
4. Use official splits, or split at document level before tokenization.
5. Begin with a fixed small number of stories.
6. Cache processed token IDs with dataset and tokenizer metadata.
7. Detect incompatible or stale cache metadata.
8. Keep network access out of ordinary unit tests.

### Required tests

- valid fake records become a correct token stream;
- missing, empty, and non-string fields follow the documented policy;
- story boundaries contain the chosen separator;
- train and validation documents do not overlap;
- cache restoration preserves tokens and metadata;
- incompatible tokenizer metadata rejects the cache.

### Done when

A small real subset produces the same batch interface already proven with local
text, so model code needs no dataset-specific changes.

## Milestone 15: Run the first online-data smoke training

### Goal

Validate the complete path from online records to a restored, generating model.

### Run plan

1. Use the smoke model configuration.
2. Use a small fixed dataset subset.
3. Print configuration, parameter count, device, and token counts.
4. Generate one untrained sample from a fixed prompt.
5. Run a small number of training steps.
6. Evaluate validation loss.
7. Save a checkpoint.
8. Reload it in a new process.
9. Generate from the same prompt.
10. Record runtime, loss, and output in an experiment note.

### Stop conditions

Stop immediately for non-finite loss, invalid token IDs, unexpectedly empty data,
checkpoint errors, or obvious split leakage.

### Done when

The end-to-end run completes quickly, the checkpoint restores, and the report
contains enough details for you to reproduce it.

### Review gate 7

Request review of the smoke-run report before increasing model size, dataset size,
or runtime.

## Milestone 16: Run a meaningful small training experiment

### Goal

Train a small model long enough to observe learned language structure.

### Before starting

Write down:

- question and hypothesis;
- exact dataset subset;
- tokenizer and vocabulary size;
- architecture and parameter count;
- batch size, context length, and effective tokens per update;
- step or token budget;
- evaluation/checkpoint frequency;
- expected output directory;
- available compute and a stopping rule.

### During training

Track training loss, validation loss, learning rate, gradient norm, tokens per
second, tokens processed, and elapsed time. Save samples from fixed prompts at
consistent checkpoints.

### After training

1. Load the best checkpoint independently.
2. Recalculate validation loss.
3. Generate fixed-prompt samples at several temperatures.
4. Compare with the bigram and untrained model.
5. Document failure modes, not only attractive samples.
6. State what the experiment does not prove.

### Done when

The report distinguishes evidence from interpretation and proposes one controlled
next experiment.

## Milestone 17: Implement BPE as a separate learning project

### Goal

Understand subword tokenization without destabilizing the known-good model path.

Do this only after character or byte training works end to end.

### Suggested components

1. Initial symbol representation.
2. Adjacent-pair frequency counting.
3. Deterministic selection of the next merge.
4. Repeated merge training to a vocabulary target.
5. Encoding with learned merge order.
6. Decoding.
7. Vocabulary and merge-rule serialization.

### Required tests

- supported input round-trips;
- pair counts match hand-worked examples;
- tie-breaking is deterministic;
- learned merges apply in correct order;
- tokenizer state restores exactly;
- validation text never influences tokenizer training.

### Controlled comparison

Hold model parameter budget, training token budget, and dataset constant as much as
possible. Report changes in sequence lengths, vocabulary-head parameters, loss,
throughput, and sample quality. Explain why raw perplexity may not be directly
comparable between tokenizers.

## Milestone 18: Create an interview-ready project report

### Goal

Turn implementation experience into explanations useful in interviews and work.

### Include

- architecture diagram;
- tensor-shape table;
- causal-mask explanation and leakage test;
- parameter-count breakdown;
- data and tokenizer decisions;
- training and validation curves;
- checkpoint/resume design;
- controlled experiment results;
- generated examples, including failures;
- compute limitations and scaling expectations;
- next improvements.

### Practice three explanation levels

For each major concept, prepare:

1. a plain-language intuition;
2. a tensor-and-equation explanation;
3. a short implementation walkthrough.

### Final demonstration

Starting from a clean environment and a small cached fixture, demonstrate:

1. installation;
2. tests;
3. a CPU smoke training run;
4. checkpoint restoration;
5. prompted generation;
6. where to find the experiment report.

## Suggested milestone commits

Use clear checkpoints such as:

```text
docs: define mini-gpt learning project
build: add importable package and test setup
feat: add character tokenizer
feat: add next-token data windows
feat: train bigram language model
feat: add evaluation and sampling
feat: implement causal attention head
feat: implement multi-head attention
feat: assemble transformer block
feat: assemble mini-gpt model
feat: add experiment configuration
feat: add checkpointed training loop
data: add validated TinyStories adapter
experiment: record TinyStories smoke run
experiment: compare small transformer with bigram
```

Commits are learning checkpoints, not quotas. Commit only a working and understood
state.

## Final dependency map

```text
environment
    -> tensor/autograd exercises
    -> tokenizer
    -> shifted data windows
    -> bigram model
    -> evaluation and generation
    -> single causal attention head
    -> multi-head attention
    -> Transformer block
    -> mini-GPT
    -> configuration
    -> robust trainer and checkpoints
    -> online dataset inspection
    -> dataset adapter
    -> online smoke run
    -> meaningful experiment
    -> tokenizer and architecture comparisons
    -> final report
```

If a later stage fails, walk backward through this graph and rerun the narrowest
relevant test. The roadmap is complete when you can rebuild and explain the path,
not merely when one trained checkpoint exists.

