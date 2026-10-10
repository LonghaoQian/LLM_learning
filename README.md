# LLM_learning

A learning project for building a small decoder-only language model (a
"mini-GPT") from scratch with PyTorch.

The goal is understanding, not state-of-the-art quality: tensors, autograd,
tokenization, causal self-attention, Transformer blocks, training, evaluation,
and text generation, each built and tested by hand.

## Approach

- Write the implementation yourself; use explicit PyTorch building blocks
  (`nn.Linear`, `nn.Embedding`, `LayerNorm`, `AdamW`, `DataLoader`).
- No pretrained weights, no `torch.nn.Transformer`, no high-level trainers.
- Introduce one concept at a time, keep every stage runnable, and back each
  concept with a small test or observable success check.
- Train on a small, locally cached subset of an online dataset (TinyStories)
  only after the model can overfit a tiny batch.

## Learning path

Start with [docs/learning-guide.md](docs/learning-guide.md), the roadmap and
index. The topic guides, in order:

| # | Guide | Deliverable |
| - | --- | --- |
| 1 | [PyTorch and neural-network basics](docs/01-pytorch-basics.md) | Tested character tokenizer and bigram language model |
| 2 | [Self-attention and Transformer theory](docs/02-self-attention.md) | Tested mini-GPT architecture with causal attention |
| 3 | [Network training and online datasets](docs/03-training-and-online-data.md) | Documented TinyStories experiment |
| – | [Detailed build roadmap](docs/04-build-roadmap.md) | File-by-file build order, tests, and review gates |

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install PyTorch using the command for your OS/accelerator from pytorch.org
python -m pip install pytest
```

Check the installation and device:

```bash
python src/let01_pytorch_learning.py
```

## Repository layout

```text
docs/      Learning guides and, later, experiment reports
src/       Implementation (grows into the src/minigpt/ package)
tests/     Fast, deterministic unit tests
```

Files are added as each learning phase begins. Datasets, checkpoints, and
experiment outputs are kept local and ignored by Git.

## Checks

```bash
pytest -q
```

## Status

Phase 1 (PyTorch basics) is in progress.
