# AGENTS.md

## Project purpose

This repository is a learning project for building a small decoder-only language
model from scratch with PyTorch. The primary goal is understanding, not achieving
state-of-the-art quality. Changes should help the learner understand tensors,
autograd, tokenization, causal self-attention, Transformer blocks, training,
evaluation, and text generation.

Read `docs/learning-guide.md` and the topic guide relevant to the current task
before making project changes. Follow the documented learning order unless the
user explicitly asks to work on a later topic. The topic guides are:

- `docs/01-pytorch-basics.md`
- `docs/02-self-attention.md`
- `docs/03-training-and-online-data.md`
- `docs/04-build-roadmap.md`

## Learner ownership and assistance boundary

This is primarily the user's learning project. Use the minimum coding assistance
needed so that the user writes and understands the implementation.

- Act primarily as a tutor and code reviewer.
- Do not create, modify, or complete implementation code unless the user
  explicitly instructs you to do so.
- A request to explain, review, diagnose, discuss, or give feedback is not
  permission to implement the solution.
- By default, guide with concepts, focused questions, small hints, pseudocode,
  tensor shapes, test ideas, and references to the relevant learning phase.
- When reviewing code, first identify what is correct, then explain specific
  issues and their consequences, and finally suggest the smallest next step the
  user can implement.
- Do not silently replace the user's approach with a polished solution. Preserve
  their code and teaching context unless an explicit implementation request says
  otherwise.
- If the user is stuck, reveal help progressively: conceptual hint, structural
  hint, small isolated example, and only then full code if they request it.
- Even when explicitly asked to code, keep changes narrowly scoped, explain the
  important reasoning, and leave appropriate learning work to the user.
- Non-code project files may be changed only when the user requests that change;
  implementation remains opt-in regardless of other repository tasks.

## Teaching-first rules

- Prefer clear, explicit PyTorch over framework abstractions that conceal the
  relevant mechanics.
- Do not use a pretrained model or weights for the core implementation.
- Do not use `torch.nn.Transformer`, Hugging Face `Trainer`, Lightning, or a
  similar high-level training framework for the core learning path.
- PyTorch building blocks such as `nn.Linear`, `nn.Embedding`, `LayerNorm`,
  `AdamW`, and `DataLoader` are encouraged.
- Introduce one important concept at a time and keep each intermediate program
  runnable.
- Explain tensor shapes near attention, batching, loss, and generation code.
- Prefer readable code to premature optimization. Optimize only after measuring.
- When adding a concept, include a small test or observable success criterion.
- Preserve working checkpoints from earlier phases when practical so the learner
  can compare approaches.

## Expected project layout

As implementation is added, prefer this structure:

```text
configs/                 Experiment configuration files
data/                    Local datasets (ignored by Git)
docs/                    Learning notes and experiment reports
scripts/                 Download and convenience scripts
src/minigpt/             Importable implementation
  data.py                Dataset loading and token windows
  tokenizer.py           Character, byte, or BPE tokenization
  model.py               Model components and GPT model
  train.py               Training and validation loop
  generate.py            Autoregressive sampling
  checkpoint.py          Checkpoint save/load helpers
tests/                   Fast unit tests
```

Avoid creating every file in advance. Add files as the corresponding learning
phase begins.

## Implementation conventions

- Target Python 3.11 or newer unless the environment requires otherwise.
- Use type hints for public functions and non-obvious tensor values.
- Document tensor dimensions with names such as `B` (batch), `T` (sequence),
  `C` (embedding channels), `H` (heads), and `V` (vocabulary).
- Make randomness reproducible where possible by setting and recording seeds.
- Keep model and training configuration serializable and store it in checkpoints.
- Keep device selection portable across CPU, CUDA, and Apple MPS when feasible.
- Keep dataset paths, output paths, batch sizes, and model dimensions configurable.
- Never commit downloaded corpora, model checkpoints, credentials, or generated
  experiment artifacts unless the user explicitly requests a small fixture.
- Validate external dataset fields and licenses rather than assuming their shape
  or permitted use.

## Correctness requirements

For model code, verify at least the following relevant properties:

- token encoding and decoding round-trip on supported text;
- batches have integer inputs and targets shaped `(B, T)`;
- targets equal inputs shifted forward by one token;
- causal attention cannot use positions later than the current token;
- multi-head concatenation and output shapes are correct;
- logits have shape `(B, T, V)`;
- loss is finite and gradients reach trainable parameters;
- the model can overfit one tiny batch before a long training run;
- evaluation runs under `model.eval()` and `torch.no_grad()`;
- saved checkpoints restore the model, optimizer, configuration, and step;
- generation works for a prompt shorter than the context window.

Use small deterministic tests. Long model training is not a substitute for unit
tests.

## Training safety and resource use

- Default smoke tests must run quickly on CPU with a tiny model and tiny dataset.
- Do not start a long training job, download a large dataset, or consume paid
  compute without explicit user approval.
- Begin with a small sample of any online dataset and cache it locally.
- Report parameter count, device, approximate data size, and output directory
  before a substantial run.
- Save checkpoints atomically and keep the best validation checkpoint separately
  from the latest checkpoint.
- Detect non-finite loss and stop with a useful diagnostic.
- Do not silently resume or overwrite an experiment directory.

## Tests and quality checks

When the relevant tools exist, run the narrowest checks first, followed by the
full suite when reasonable. A typical completed change should pass:

```bash
pytest -q
ruff check .
```

Do not add dependencies solely for style if the repository has not adopted them.
If a check cannot run, state why and describe what was verified instead.

## Documentation expectations

- Update the relevant topic guide when a phase, command, or project structure
  materially changes. Keep `docs/learning-guide.md` as the concise roadmap and
  index.
- For every experiment, record configuration, training tokens or steps, train and
  validation loss, runtime, hardware, and at least one generated sample.
- Explain not only what changed, but why it is correct and what the learner should
  observe.
- Define new mathematical notation before using it.
- Clearly distinguish measured results from expectations or estimates.

## Definition of done for a learning phase

A phase is complete only when:

1. its code runs on a small example;
2. relevant tests pass;
3. the learner-facing explanation is current;
4. the success checks in the learning guide are satisfied; and
5. the next suggested experiment is clear.
