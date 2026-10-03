# AMD dual-GPU tuning guide

This documents the tuning process used for a 2x AMD RX 7900 XTX (`gfx1100`)
Strata dual-GPU setup, and provides a helper for reproducing the most important
step on another AMD machine.

The biggest measured win was generating a local hipBLASLt tuning table for the
exact GPU architecture and installed hipBLASLt version. On the tested rig, the
shipped tables did not match ROCm's hipBLASLt `1.4.1`, so Strata used the
untuned dense GEMM path.

## Helper

Use:

```bash
./tools/tune_strata_amd.sh --config strata-iq3_s.json
```

Common examples:

```bash
# Generate a tuning table for the detected AMD GPU and installed hipBLASLt.
./tools/tune_strata_amd.sh --config strata-iq3_s.json

# Generate and point the Strata config at the new table.
./tools/tune_strata_amd.sh --config strata-iq3_s.json --apply-config

# Generate, apply, then benchmark through an OpenAI-compatible endpoint.
./tools/tune_strata_amd.sh \
  --config strata-iq3_s.json \
  --apply-config \
  --benchmark \
  --base-url http://localhost:8080/v1 \
  --model local-ai \
  --tokenizer /path/to/tokenizer \
  --runs 5 \
  --depth 0 16384
```

The helper:

1. detects the AMD architecture, for example `gfx1100`;
2. detects the installed hipBLASLt version from `/opt/rocm/include/hipblaslt/hipblaslt-version.h`;
3. builds `build-hip/tune_hipblaslt` if needed;
4. chooses a nearby shipped case list for the same architecture;
5. runs `tune_hipblaslt` to produce:

   ```text
   tools/hip/<arch>-hipblaslt-<version>.txt
   ```

6. optionally writes `STRATA_HIPBLASLT_TUNING` into the Strata JSON config;
7. optionally runs `llama-benchy`.

After changing the config, restart the Strata target so the new environment
variable is loaded.

## Manual hipBLASLt tuning

On the tested machine:

```bash
cd /home/gysmo/Strata-DualGPU

cmake --build build-hip --target tune_hipblaslt

CASES=$(awk 'NR>2 {printf " --case %s,%s,%s,%s,%s", $1, $5, $2, $3, $4}' \
  tools/hip/gfx1100-hipblaslt-100200.txt)

./build-hip/tune_hipblaslt $CASES \
  --tuning-out tools/hip/gfx1100-hipblaslt-100401.txt
```

Then add to the Strata config:

```json
{
  "env": {
    "STRATA_HIPBLASLT_TUNING": "/home/gysmo/Strata-DualGPU/tools/hip/gfx1100-hipblaslt-100401.txt"
  }
}
```

The table filename encodes the architecture and hipBLASLt version. Do not reuse
a table across different architectures or hipBLASLt versions.

## Benchmark command

The tuning sweep used:

```bash
llama-benchy \
  --base-url http://localhost:8080/v1 \
  --model local-ai \
  --tokenizer /home/gysmo/Strata-data/packs/iq3_s/tokenizer \
  --pp 512 \
  --tg 512 \
  --depth 0 16384 \
  --concurrency 1 \
  --runs 5 \
  --format md
```

Use the same tokenizer fallback for every run. On this setup the Strata
tokenizer directory was not directly loadable by `llama-benchy`, so it fell
back to `gpt2`; that is acceptable for relative comparisons as long as every
run uses the same fallback.

## Tuning steps tried

### 1. Baseline dual-GPU config

The dual-GPU target used:

```text
--expert-cache auto
--prefill auto
--spec 4
--spec-min-p 0.5
--kv int8
--kv-resident 32768
--layer-split auto
```

The first useful AMD-specific change was:

```text
--pcie-frac 0
```

This disables the GPU-side PCIe expert share. On the tested 2x RX 7900 XTX rig,
this improved decode and matched the AMD guidance in `docs/AMD_HIP.md`.

### 2. Request-level `pcie_frac` sweep

Before restarting the engine, `pcie_frac` can be swept per request:

```bash
llama-benchy ... --extra-body 'strata_tune={"pcie_frac":0.0}'
llama-benchy ... --extra-body 'strata_tune={"pcie_frac":0.2}'
```

Result on the tested rig:

| `pcie_frac` | tg512 t/s |
|---:|---:|
| auto | 94.31 ± 13.98 |
| `0.0` | 98.47 ± 11.44 |
| `0.2` | 89.35 ± 6.70 |

Conclusion: use `--pcie-frac 0` for this AMD rig.

### 3. Request-level `spec_min_p` sweep

```bash
llama-benchy ... --extra-body 'strata_tune={"spec_min_p":0.45}'
llama-benchy ... --extra-body 'strata_tune={"spec_min_p":0.55}'
llama-benchy ... --extra-body 'strata_tune={"spec_min_p":0.60}'
llama-benchy ... --extra-body 'strata_tune={"spec_min_p":0.65}'
llama-benchy ... --extra-body 'strata_tune={"spec_min_p":0.70}'
```

No stable win was found. The default `--spec-min-p 0.5` remained better overall.

### 4. `setup.sh --calibrate`

```bash
./setup.sh --calibrate --no-start --yes
```

On the tested AMD rig, calibration selected:

```text
--pcie-frac 0.75
--spec-min-p 0.70
```

That improved long-context slightly but hurt short-context decode:

| Config | tg512 @ depth 0 | tg512 @ depth 16384 |
|---|---:|---:|
| baseline with `pcie_frac=0` | 110.61 ± 15.09 | 79.37 ± 8.17 |
| calibrated | 92.01 ± 6.87 | 83.01 ± 11.08 |

Conclusion: run calibration, but inspect the chosen settings. Do not blindly
keep them for AMD dual-GPU decode.

### 5. Layer split sweep

The automatic split chose `K=29`, meaning:

```text
GPU0: layers 0-28
GPU1: layers 29-47
```

Swept explicit splits:

| `layer_split` | tg512 @ depth 0 | tg512 @ depth 16384 |
|---:|---:|---:|
| auto | 110.61 ± 15.09 | 79.37 ± 8.17 |
| 24 | 103.02 ± 18.76 | 76.67 ± 6.26 |
| 29 | 95.22 ± 18.24 | 89.95 ± 9.20 |
| 34 | 76.24 ± 13.19 | 83.89 ± 10.28 |

Conclusion:

- `auto` is best for short-context throughput;
- explicit `29` can be better for long-context decode.

### 6. Pipeline windows

The dual-GPU fork defaults to pipelined windows on two GPUs.

| `pipeline-windows` | tg512 @ depth 0 | tg512 @ depth 16384 |
|---:|---:|---:|
| default, `2` | 110.61 ± 15.09 | 79.37 ± 8.17 |
| `0` | 98.36 ± 8.28 | 77.23 ± 5.52 |
| `1` | 82.66 ± 10.49 | 62.39 ± 4.67 |

Conclusion: keep the default pipelined mode.

### 7. `adapt-every=0`

```text
--adapt-every 0
```

Result:

| Config | tg512 @ depth 0 | tg512 @ depth 16384 |
|---|---:|---:|
| baseline | 110.61 ± 15.09 | 79.37 ± 8.17 |
| `adapt-every=0` | 80.52 ± 16.28 | 77.09 ± 3.61 |

Conclusion: worse on this rig.

### 8. hipBLASLt tuning table

This was the largest win.

| Config | tg512 @ depth 0 | tg512 @ depth 16384 |
|---|---:|---:|
| baseline with `pcie_frac=0` | 110.61 ± 15.09 | 79.37 ± 8.17 |
| hipBLASLt table | 115.79 ± 12.92 | 90.55 ± 8.58 |
| hipBLASLt table + `expert_profile_save` learning run | 117.60 ± 11.16 | 92.80 ± 7.27 |
| hipBLASLt table + `layer_split=29` | 106.89 ± 11.81 | 93.27 ± 7.83 |

Conclusion:

- default recommendation: hipBLASLt table + `pcie_frac=0` + `layer_split=auto`;
- long-context alternative: hipBLASLt table + `layer_split=29`.

### 9. Expert profile saving

Added:

```json
{
  "expert_profile_save": "data/expert-profile-learned.bin"
}
```

The benchmark run with this enabled was fast, but the learned profile file was
not created during the short test window. The engine may save it after a clean
exit or after a longer idle interval. Keep the setting if you want the engine to
learn over time, but do not assume the file appears immediately.

## Recommended config for 2x RX 7900 XTX

```json
{
  "args": [
    "--pcie-frac",
    "0"
  ],
  "env": {
    "STRATA_HIPBLASLT_TUNING": "tools/hip/gfx1100-hipblaslt-100401.txt"
  },
  "expert_profile_save": "data/expert-profile-learned.bin",
  "layer_split": "auto"
}
```

For long-context-heavy workloads, try:

```json
{
  "layer_split": "29"
}
```

## Notes for other architectures

- Use the helper to generate a table for your architecture and hipBLASLt
  version.
- If no shipped case list exists for your architecture, copy the closest case
  list and review the GEMM shapes. The shapes should match the engine's dense
  projection shapes.
- Re-run the benchmark after every config change.
- Restart the Strata target after changing config args or environment variables.
- Keep `pcie_frac` as a first experiment on AMD rigs. On some AMD cards,
  `0` is better than the NVIDIA-oriented default.
