# Small Language Model Learning Roadmap

This project teaches you to build a small decoder-only language model with
PyTorch. The material is divided into three focused guides so implementation,
theory, and training practice do not become mixed together.

The objective is understanding, not reproducing a production-scale LLM. You
should write the implementation yourself, use the coding assistant mainly for
explanations and reviews, and move forward only when you can explain the current
stage.

## Study order

### 1. [PyTorch and neural-network basics](01-pytorch-basics.md)

Start here. Learn tensors, shapes, batches, modules, autograd, loss functions,
optimizers, datasets, and the basic training loop. Finish by building a character
tokenizer and a bigram language model.

Main outcome: you can explain and debug a small neural-network training loop.

### 2. [Self-attention and Transformer theory](02-self-attention.md)

Study queries, keys, values, scaled dot-product attention, causal masking,
multi-head attention, positional information, residual connections, and
Transformer blocks.

Main outcome: you can derive the attention shapes and implement causal attention
using ordinary PyTorch tensor operations.

### 3. [Network training and online datasets](03-training-and-online-data.md)

Turn the model into a reliable experiment. Learn validation, AdamW, learning-rate
schedules, gradient clipping, checkpointing, generation, online dataset loading,
tokenizer choices, evaluation, and debugging.

Main outcome: you can train a small GPT-style model on a controlled TinyStories
subset and explain the results.

### Implementation companion: [Detailed build roadmap](04-build-roadmap.md)

Use this while working through the three guides. It gives the exact order in
which to create project files and features, the tests to write at every step,
review gates, and the observable result required before continuing.

## How to use each guide

For each section:

1. Read the concepts and predict what the code will need to do.
2. Complete the exercises without asking for a full solution first.
3. Inspect tensor shapes and intermediate values.
4. Write the suggested tests.
5. Run the success checks.
6. Explain the interview questions aloud in your own words.
7. Ask for a code review before moving to the next major component.

When stuck, ask for help progressively:

1. a conceptual hint;
2. a diagram or shape walkthrough;
3. pseudocode;
4. review of your attempted code;
5. a small isolated example;
6. full implementation only when you explicitly want it.

## Suggested pace

At roughly 5–8 focused hours per week:

| Guide | Suggested time | Deliverable |
| --- | ---: | --- |
| PyTorch basics | 2–3 weeks | Tested bigram language model |
| Self-attention | 2–3 weeks | Tested mini-GPT architecture |
| Training and data | 2–4 weeks | Documented TinyStories experiment |

This schedule is deliberately flexible. Understanding one small working model is
more valuable than rapidly assembling a large model from copied code.

## Completion checklist

- [ ] I can explain tensor shape, dtype, and device.
- [ ] I can write and explain a complete optimization loop.
- [ ] My tokenizer and shifted input/target batches pass tests.
- [ ] My bigram baseline trains and generates text.
- [ ] I can derive the shapes of Q, K, V, attention scores, and output.
- [ ] My causal-attention test proves that future tokens cannot leak backward.
- [ ] My mini-GPT can overfit one small batch.
- [ ] Training and validation are measured separately.
- [ ] Checkpoints restore model, optimizer, configuration, and training step.
- [ ] I can train on a small online dataset subset reproducibly.
- [ ] I record configuration, hardware, losses, and generated samples.
- [ ] I can explain what failed and what I would test next.

## Reference conventions

The guides use these dimension names:

```text
B = batch size
T = sequence length
C = embedding/channel size
H = number of attention heads
D = size of one attention head, usually C / H
V = vocabulary size
```

Project implementation should remain small and explicit. Use PyTorch components
such as `nn.Linear`, `nn.Embedding`, `LayerNorm`, `AdamW`, and `DataLoader`, but do
not hide the core model behind `torch.nn.Transformer`, a pretrained model, or a
high-level trainer.
