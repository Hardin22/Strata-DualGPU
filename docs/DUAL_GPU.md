# Two GPUs and 32 GB of RAM

This is what the fork changes, why, and what each change measured. Everything was measured on one PC:

- RTX 5080 16 GB (drives two 4K monitors) and RTX 4060 Ti 16 GB (on a PCIe 4.0 x4 chipset slot)
- i9-14900KF (8 P-cores, 16 E-cores), 32 GB of RAM, Windows 11
- Swift 1.5 Qwen3.8-Flash-Next GSQ-RCO IQ2_XS, 256K context, Q4_0 KV cache, `--spec 4`

The benchmark is the same every time: a fixed set of code and prose prompts, greedy, 500 tokens, plus a 32K-token
prompt. "Code" and "prose" below are decode speeds in tokens per second.

## The starting point

On a 32 GB PC the model's experts do not fit in RAM next to everything else, so Strata uses its low-RAM mode: the
GPU cache holds the most-used experts and the rest are copied into RAM once (`--resident-experts`). Upstream only
does that on one card. With a layer split it falls back to reading the experts through the OS file cache, which
on 32 GB means reading them from the SSD while it answers. So setup recommends one GPU, and the second card sits idle.

Two 16 GB cards hold about twice the experts one card holds. Getting them to work with the RAM copy was the first
step; the rest of the work was about the fact that, on a split, the cards take turns.

## 1. The resident RAM mode on a layer split

`--resident-experts` now works with `--layer-split`. Each stage's GPU cache is left out of the RAM copy, and what
remains is copied hottest first until the RAM budget runs out (upstream's #467 path, which already handles a copy
that does not fit). With 2 x 16 GB the caches hold about 13,000 experts and the RAM copy the other ~11,000
(15.5 GiB here), so nothing is read from the SSD while it answers.

One bug had to be fixed on the way: an adaptive swap copies the evicted expert from VRAM back into RAM, and with a
split it read the slot number of a later stage's cache out of CUDA0's cache. The RAM copy then held another
expert's weights and the output turned to garbage after the first few swaps. A short benchmark does not catch
this; a 30-request stress run does.

`--vram-reserve-later-mib` gives the later cards their own VRAM reserve. The card that drives the monitors needs
more headroom than the other one.

## 2. Card order: the faster card last

The last stage also runs the output head, the draft layer and (with pipelining, below) the draft chain that the
next window waits for. Putting the 5080 last and the 4060 Ti first was worth about 12% on its own (code 85-90 to
96-101 tok/s at the time). Upstream's docs recommend the opposite order.

The server now orders the cards by speed (SMs x clock, read from `strata --list-gpus`), fastest last, when the
config's `layer_split` is `auto`. `"gpu_order": "as_given"` keeps your order. An explicit split such as `"20"` also
keeps it, because the number was picked for that order.

## 3. The adaptive expert tier no longer stalls a window

Upstream's adaptive tier swaps experts between VRAM and RAM in rounds: pick the swaps, copy the evicted experts back
with a stream sync, copy the new ones in from pageable RAM. With ~90 swaps a round took ~24 ms, every 4th window.

`--adapt-async 1` (the default) splits a round in four steps that run on a job thread and advance between windows,
so no window waits for one. A slot is overwritten only after its old expert stopped being planned for the GPU, and
a RAM place only after its expert is resident. The same change made the CPU side of a window cheaper (an AVX2
activation quantizer, no host zeroing of rows the GPU fills anyway, no PCIe path recorded in a stage that never uses
one).

## 4. Fewer host waits per window

On a split the host used to synchronize stage 0 before launching stage 1, and paid for the commit and the draft
round one after the other. Now:

- the stages hand off through a flag in mapped pinned memory that the next stage's graph waits on, so both graphs
  are launched at once (a cross-GPU round trip through mapped memory measured 7-10 us here)
- commits are queued on each stage's stream and waited for once per request
- the MTP draft round is a single CUDA graph, with each chain step behind a conditional node that the GPU evaluates
  with the same min-p rule the host used
- the draft round is launched before the window's tokens are emitted

On the 0.1.30 base, sections 3 and 4 together with 15 pool workers (section 7) and a draft min-p of 0.65 took code
from 89.6 to 117.5 tok/s and prose from 70.3 to 86.3. Measured alone, the host-wait changes were worth code 95 to
108 and prose 75 to 85.

## 5. Dense kernels

A decode window is a long chain of small kernels, and on the 5080 many of them finish before the next one has its
weights. Programmatic dependent launch (sm_90 and newer) lets each kernel start fetching its weights into L2 while the
previous one finishes; the query side of attention runs on a side stream beside the K/V side; a Q4_0 KV cache, the
PLE post-ops and the split's hand-off copies are batched over the window. Every change keeps the arithmetic of the
path it replaces, checked with per-layer hashes over whole requests. Code 117.5 to 131.8 tok/s, prose 86.3 to 97.1
(with the hyper-connection read of the next paragraph included).

Upstream later shipped its own staged hyper-connection read (#315) and decode-once IQ kernels (#242), which did what
two parts of this work did, so the fork uses upstream's.

## 6. Pipelined windows: both cards at once

Even with all of the above, a profile of one window showed each card idle for more than half of it: the 4060 Ti ran
its 20 layers (~14 ms), then waited while the 5080 ran its 28 layers, the head and the draft (~12 ms), and the
other way round.

`--pipeline-windows 2` runs window K+1 on stage 0 while stage 1 still verifies window K. The tokens for K+1 come from
a teacher-forced MTP chain that starts as soon as K's first outputs land. If K accepts what the chain assumed, K+1
is already half done. If it does not, stage 0 restores its DeltaNet state from a snapshot taken on a side stream and
K+1 is rebuilt from the real tokens. A gate skips the speculative launch when the chain's own confidence says it is
unlikely to be kept.

Same server, serial vs pipelined:

| | code | prose | thinking (T=1.0) | copy-heavy edit |
|---|---:|---:|---:|---:|
| serial | 116.0 | 87.8 | 84.4 | 136.1 |
| pipelined | 136.1 | 96.0 | 89.2 | 177.3 |
| | x1.17 | x1.09 | x1.06 | x1.30 |

Edits gain the most because a prompt-lookup window (text copied from the context) chains straight into the next
speculative window.

It is on by default when the split runs on exactly two GPUs. `--pipeline-windows 0` turns it off. Requests with
repetition penalties or coupled drafting use the serial loop.

The automatic split (`--layer-split auto`) had to learn about it. Upstream's search adds the stages' times up, which
is right when they take turns: on this PC it put 2 of the 48 layers on the 4060 Ti and decoded code at 29 tok/s. With
pipelining a window costs about the slower stage, and a stage's CPU misses wait on that stage, so the search now
prices each stage by its layers, its misses and (on the last card) the head and the draft chain, and takes the
slower one. It picks K=19 here: 158 tok/s on code against 165 for the hand-tuned K=20.

## 7. CPU and OS defaults

- On a hybrid Intel CPU the expert pool now uses the P-cores (minus the one the host loop spins on) plus half of
  the E-cores. An E-core runs the expert kernels about 2.2x slower than a P-core and a layer waits for its slowest
  part. On the 14900KF that is 15 workers instead of 23, and with pipelining it made the biggest difference of any
  default: code 106 to 165 tok/s, prose 84 to 116. `--pool-workers N` overrides it.
- On Windows the engine opts out of power throttling (EcoQoS), which can run a background process's threads at
  efficiency clocks. `STRATA_NO_ECOQOS=0` leaves it to Windows.

## 8. VRAM reserves

When the config gives no reserve, the server reserves 1800 MiB on a card that drives a display and 300 MiB on one
that drives none. 1800 MiB is what two 4K monitors, a browser and the compositor needed here without the driver
running out of VRAM. With one 1080p monitor you can go lower with `--vram-reserve-mib` (first card) and
`--vram-reserve-later-mib` (the others). `"vram_reserve": "as_given"` in the config turns the automatic choice off.

## 9. Each card keeps only its own layers' weights

A split used to load the dense weights of all 48 layers on every card, although each card runs only its own layers.
Now each card keeps the weights of its layers, and the VRAM this frees goes to its expert cache. Upstream has the same
idea in two open pull requests, #559 (blange48) and #639 (JeanP00l), for explicit split points. Here it also works
with `--layer-split auto`: every card loads everything first, the search counts what trimming will free on each card
when it places the boundary, and then each card reloads only its own layers before the sessions and caches are made.

On this PC that frees 1.9 GiB on the 4060 Ti and 1.3 GiB on the 5080. Same engine, `--no-trim-stage-weights`
against the default, K=20:

| | code | prose | code, 32K context | prose, 32K context | 32K prompt |
|---|---:|---:|---:|---:|---:|
| IQ2_XS, every card holds every layer | 162 | 109 | 149 | 100 | 1,899 |
| IQ2_XS, own layers only | 172 | 111 | 157 | 104 | 2,201 |
| IQ3_XXS (160K context), every layer | 141 | 95 | 98 | 71 | 1,006 |
| IQ3_XXS, own layers only | 144 | 97 | 131 | 92 | 1,978 |

IQ3_XXS gains the most. Its experts are bigger, and before this change the ones no card held did not all fit in RAM,
so about 2,400 of them were read from the SSD in one benchmark run. With 2,174 more of them in VRAM the rest fit, and
none are read from the SSD. The automatic split still picks K=20 for it.

More experts in VRAM means more of them computed by the GPU, which rounds differently from the CPU (see
[Is the output still the same model?](#is-the-output-still-the-same-model)). On the 5,335 tokens of that comparison, IQ3_XXS with and without trimming picked the same top token at 93.5%
of the positions, with a perplexity of 6.95 and 7.03.

## Where it ended up

The stock draft head throughout, decode in tokens per second:

| | code | prose | 32K prompt |
|---|---:|---:|---:|
| 0.1.30 with sections 1 and 2 | 89.6 | 70.3 | |
| + sections 3, 4 and 7 | 117.5 | 86.3 | 1634 |
| + section 5 | 131.8 | 97.1 | 1618 |
| + section 6 | 145.3 | 101.6 | |
| all of it rebased on 0.1.38 (this repo) | 143.4 | 102.3 | 1940 |

## Is the output still the same model?

Not bit for bit, and it never was across settings: a token's CPU expert rows round differently when they are
computed alone or in a group, so the text depends on how the drafts cut it into windows, and an expert computed on
the CPU rounds differently from the same expert on the GPU. Anything that changes which experts sit in VRAM changes
the last bits, and at a near-tie that picks the other token. Upstream documents the same thing.

To check that nothing worse happened, I compared the next-token distributions of upstream 0.1.38 and this fork on
the same 5,333 tokens (a Python file, an English doc and an Italian text, teacher-forced through the verify windows
with `STRATA_LOGPOS`; upstream ran the same split, from the file cache). The end-of-turn token a chat model
expects inside a user message is taken out of each distribution, as in upstream's own comparison with llama.cpp.

| | upstream 0.1.38 | this fork | this fork, 1 GB smaller cache on the 5080 |
|---|---:|---:|---:|
| perplexity | 7.28 | 7.24 | 7.26 |
| top-1 = the real next token | 64.1% | 64.0% | 63.9% |

Upstream and the fork pick the same top token at 93.9% of the positions; the fork against itself with a smaller
cache, at 96.3%. Most of the disagreements are at near-ties.

With `STRATA_IQ_MT_MIN=1 --adapt-every 0` the output no longer depends on the window cuts. On one server (so the
same experts in VRAM) the pipelined loop then wrote the same text as the serial one in 45 of 45 requests: greedy,
seeded sampling, and runs with forced rollbacks.

## Settings and switches

| | |
|---|---|
| `--resident-experts` with `--layer-split` | the RAM copy across a split (section 1) |
| `--vram-reserve-later-mib N` | VRAM left free on the later cards of a split |
| `--pipeline-windows 0/1/2` | pipelined windows; 2 by default on a two-GPU split |
| `--layer-split auto` | balances the two stages when pipelined (section 6) |
| `--no-trim-stage-weights` | every card keeps the dense weights of all the layers (section 9) |
| `--adapt-async 0` | the blocking adaptive tier |
| `--list-gpus` | prints the visible cards and exits |
| `"gpu_order": "as_given"` | the server keeps the config's card order |
| `"vram_reserve": "as_given"` | the server adds no reserves |
| `STRATA_XSTAGE=0` | the host synchronizes each stage instead of the device flags |
| `STRATA_COMMIT_ASYNC=0` | wait for each commit |
| `STRATA_MTP_CHAIN=0` | one graph launch per draft step |
| `STRATA_DF_PDL=0` | no programmatic dependent launch |
| `STRATA_DF_BRANCH=0` | no side streams inside the verify graph |
| `STRATA_NO_ECOQOS=0` | leave Windows power throttling on |
| `STRATA_HOST_CPU=n` | pin the host loop to logical CPU n |
| `STRATA_DECODE_TIMING=1` | per-window host and per-stage GPU timings in the engine log |
| `STRATA_PIPELINE_TRACE=<file>` | a timeline of the pipelined loop |
