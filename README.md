# Strata, two-GPU edition

A fork of [Strata](https://github.com/Niko1221/Strata) for a PC with two NVIDIA cards and 32 GB of RAM. I run
Qwen3.8-Flash-Next with it every day (the Swift 1.5 IQ2_XS fine-tune) behind a coding agent, on an RTX 5080 and an
RTX 4060 Ti.

Same PC, same model, same settings (256K context):

| | writes code | writes prose | reads a 32K prompt |
|---|---:|---:|---:|
| Strata 0.1.38 on the 5080 alone (what its setup picks for 32 GB) | 29 tok/s | 28 | 333 |
| this fork on both cards | 143 tok/s | 102 | 1,940 |
| this fork, with my fine-tuned draft head | 161 tok/s | 105 | 1,854 |

With Pi, the coding agent I use, it peaks at 209 tokens/s, rarely drops under 100, and reads a 150K-token
context at about 1,800 tokens/s. That is with the fine-tuned draft head, which is not in this repo yet.

## Why a fork

This model's experts don't fit in 32 GB of RAM next to everything else, so Strata uses its low-RAM mode: the
GPU keeps the most-used experts and the rest are copied into RAM. Upstream does that on one card only. If you split
the layers across two cards, it reads those experts from the SSD instead, so its setup tells you to use one GPU and
the second card does nothing.

Here the RAM mode works on a split. And because the two cards of a split take turns (each sat idle for more than
half of every step on my PC), most of the work went into making them run at the same time.

## What changed

- The RAM copy of the experts works with `--layer-split`: each card caches the hottest experts of its own layers,
  RAM holds the rest, nothing comes from the SSD.
- Pipelined windows: the first card starts the next decoding step while the second is still checking the current
  one. On by default with two GPUs.
- The faster card goes last, since it also runs the output head and the draft model, and the automatic split
  balances the two cards for pipelining. The server orders the cards by itself.
- Adaptive expert swaps between VRAM and RAM no longer stop decoding while they copy.
- Fewer host waits and kernel launches per step: device-side flags between the cards, the draft chain in one CUDA
  graph, programmatic dependent launch on RTX 50 cards.
- Defaults for the rest of the PC: on an Intel hybrid CPU the expert pool runs on the P-cores and half of the
  E-cores, and the card that drives your monitors keeps more VRAM free.

What each change measured, and every switch: [docs/DUAL_GPU.md](docs/DUAL_GPU.md).

The answers are as good as upstream's. On 5,333 tokens of code, English and Italian, read the same way by both
engines, the perplexity is 7.24 here and 7.28 upstream, and both pick the right next token 64% of the time.

## Install

The same as Strata: download this repo, run `START-HERE.bat` (Linux: `./setup.sh`) and press Enter at every
question. With two cards and 32 GB it now sets up both cards in the low-RAM mode. The ready-made Windows engine comes
from this repo's releases; on Linux setup compiles it.

Already using Strata? Unzip this next to it and run `START-HERE.bat`: Strata keeps the models in a data folder
shared by every copy on the PC, so nothing is downloaded again. A model set up for one card asks once whether to use
both. With `"layer_split": "auto"` in the config (the default), the server picks the card order and the VRAM
reserves at every start.

## What I tested

One PC: RTX 5080 + RTX 4060 Ti (PCIe 4.0 x4), i9-14900KF, 32 GB, Windows 11, the Swift 1.5 IQ2_XS model. The other
models and cards go through the same code, but I haven't run them. I haven't built it on Linux or for AMD cards.
The pipelining needs exactly two cards; with three, the split runs without it.

Like upstream with its adaptive expert swaps on, the same prompt can give slightly different text from one run to
the next, at near-ties between two tokens.

## Upstream

Everything else (the models, setup, the web app, the API and how the engine works) is Strata by Niko1221 and its
contributors, MIT licensed like this fork: their README is [README.upstream.md](README.upstream.md), and the docs
folder is theirs apart from DUAL_GPU.md. This fork is based on Strata 0.1.38.
