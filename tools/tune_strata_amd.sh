#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  tools/tune_strata_amd.sh [options]

Generates a hipBLASLt tuning table for the detected AMD GPU and installed
hipBLASLt version. Optionally applies the table to a Strata JSON config and
runs llama-benchy.

Options:
  --config PATH              Strata JSON config. Default: strata-iq3_s.json
  --arch NAME                GPU architecture, e.g. gfx1100. Default: auto
  --hipblaslt-version N      hipBLASLt version as NNNNNN, e.g. 100401. Default: auto
  --rocm PATH                ROCm prefix. Default: /opt/rocm
  --build-dir PATH           CMake build directory. Default: build-hip
  --source-cases PATH        Case-list table used as input shapes. Default: newest matching tools/hip/<arch>-hipblaslt-*.txt
  --out PATH                 Output tuning table. Default: tools/hip/<arch>-hipblaslt-<version>.txt
  --apply-config             Write STRATA_HIPBLASLT_TUNING into --config
  --benchmark                Run llama-benchy after generating the table
  --base-url URL             llama-benchy base URL. Required with --benchmark
  --model NAME               llama-benchy model. Default: local-ai
  --tokenizer PATH           Tokenizer path for llama-benchy. Required with --benchmark
  --runs N                   llama-benchy runs. Default: 5
  --pp N                     llama-benchy pp. Default: 512
  --tg N                     llama-benchy tg. Default: 512
  --depth N [N...]           llama-benchy depths. Default: 0 16384
  --bench-bin PATH           llama-benchy binary. Default: auto-detect, else pipx run
  --dry-run                  Print commands without running them
  -h, --help                 Show this help
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
config="strata-iq3_s.json"
arch="auto"
hipblaslt_version="auto"
rocm="/opt/rocm"
build_dir="build-hip"
source_cases=""
out=""
apply_config=0
benchmark=0
base_url=""
model="local-ai"
tokenizer=""
runs="5"
pp="512"
tg="512"
depths=("0" "16384")
bench_bin=""
dry_run=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) config="${2:?missing value for --config}"; shift 2 ;;
    --arch) arch="${2:?missing value for --arch}"; shift 2 ;;
    --hipblaslt-version) hipblaslt-version="${2:?missing value for --hipblaslt-version}"; shift 2 ;;
    --rocm) rocm="${2:?missing value for --rocm}"; shift 2 ;;
    --build-dir) build_dir="${2:?missing value for --build-dir}"; shift 2 ;;
    --source-cases) source_cases="${2:?missing value for --source-cases}"; shift 2 ;;
    --out) out="${2:?missing value for --out}"; shift 2 ;;
    --apply-config) apply_config=1; shift ;;
    --benchmark) benchmark=1; shift ;;
    --base-url) base_url="${2:?missing value for --base-url}"; shift 2 ;;
    --model) model="${2:?missing value for --model}"; shift 2 ;;
    --tokenizer) tokenizer="${2:?missing value for --tokenizer}"; shift 2 ;;
    --runs) runs="${2:?missing value for --runs}"; shift 2 ;;
    --pp) pp="${2:?missing value for --pp}"; shift 2 ;;
    --tg) tg="${2:?missing value for --tg}"; shift 2 ;;
    --depth)
      depths=()
      shift
      while [[ $# -gt 0 && "${1:0:2}" != "--" ]]; do
        depths+=("$1")
        shift
      done
      ;;
    --bench-bin) bench_bin="${2:?missing value for --bench-bin}"; shift 2 ;;
    --dry-run) dry_run=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage; die "unknown option: $1" ;;
  esac
done

run() {
  if [[ "$dry_run" == "1" ]]; then
    printf 'DRY-RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

cd "$repo_root"

if [[ "$arch" == "auto" ]]; then
  if command -v rocminfo >/dev/null 2>&1; then
    arch="$(rocminfo 2>/dev/null | awk '/^gfx/{print $NF; exit}')"
  fi
  if [[ -z "$arch" ]]; then
    for f in /sys/bus/pci/devices/*/uevent; do
      if grep -q 'amdgpu' "$f" 2>/dev/null; then
        arch="$(grep -m1 -oE 'gfx[0-9a-f]+' "$f" 2>/dev/null || true)"
        [[ -n "$arch" ]] && break
      fi
    done
  fi
fi

[[ -n "$arch" ]] || die "could not detect AMD GPU architecture; pass --arch, e.g. --arch gfx1100"

if [[ "$hipblaslt_version" == "auto" ]]; then
  version_header="$rocm/include/hipblaslt/hipblaslt-version.h"
  [[ -r "$version_header" ]] || die "cannot read $version_header; pass --hipblaslt-version"
  major="$(sed -n 's/^#define HIPBLASLT_VERSION_MAJOR[[:space:]]\+\([0-9]\+\).*/\1/p' "$version_header")"
  minor="$(sed -n 's/^#define HIPBLASLT_VERSION_MINOR[[:space:]]\+\([0-9]\+\).*/\1/p' "$version_header")"
  patch="$(sed -n 's/^#define HIPBLASLT_VERSION_PATCH[[:space:]]\+\([0-9]\+\).*/\1/p' "$version_header")"
  [[ -n "$major" && -n "$minor" && -n "$patch" ]] || die "could not parse hipBLASLt version from $version_header"
  hipblaslt_version="$((major * 100000 + minor * 100 + patch))"
fi

if [[ -z "$out" ]]; then
  out="tools/hip/${arch}-hipblaslt-${hipblaslt_version}.txt"
fi

if [[ -z "$source_cases" ]]; then
  mapfile -t candidates < <(find tools/hip -maxdepth 1 -type f -name "${arch}-hipblaslt-*.txt" | sort)
  if [[ ${#candidates[@]} -gt 0 ]]; then
    source_cases="${candidates[-1]}"
  elif [[ -f tools/hip/gfx1100-hipblaslt-100200.txt ]]; then
    source_cases="tools/hip/gfx1100-hipblaslt-100200.txt"
    echo "warning: no case list for $arch; using $source_cases"
  else
    die "no source case list found; pass --source-cases"
  fi
fi

[[ -f "$source_cases" ]] || die "source case list not found: $source_cases"

run cmake --build "$build_dir" --target tune_hipblaslt

mapfile -t cases < <(awk 'NR>2 {printf "%s,%s,%s,%s,%s\n", $1, $5, $2, $3, $4}' "$source_cases")
[[ ${#cases[@]} -gt 0 ]] || die "no cases parsed from $source_cases"

tuner="$build_dir/tune_hipblaslt"
cmd=("$tuner")
for c in "${cases[@]}"; do
  cmd+=(--case "$c")
done
cmd+=(--tuning-out "$out")

run "${cmd[@]}"

if [[ "$apply_config" == "1" ]]; then
  [[ -f "$config" ]] || die "config not found: $config"
  run python3 - "$config" "$out" <<'PY'
import json
import sys
from pathlib import Path

cfg_path = Path(sys.argv[1])
table_path = Path(sys.argv[2]).resolve()
cfg = json.loads(cfg_path.read_text(encoding="utf-8-sig"))
cfg.setdefault("env", {})["STRATA_HIPBLASLT_TUNING"] = str(table_path)
cfg_path.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
print(f"updated {cfg_path}: STRATA_HIPBLASLT_TUNING={table_path}")
PY
  echo "Restart the Strata target before benchmarking the applied config."
fi

if [[ "$benchmark" == "1" ]]; then
  [[ -n "$base_url" ]] || die "--base-url is required with --benchmark"
  [[ -n "$tokenizer" ]] || die "--tokenizer is required with --benchmark"
  if [[ -z "$bench_bin" ]]; then
    if command -v llama-benchy >/dev/null 2>&1; then
      bench_bin="llama-benchy"
    elif command -v pipx >/dev/null 2>&1; then
      bench_bin="pipx run llama-benchy"
    else
      die "llama-benchy not found; install it or pass --bench-bin"
    fi
  fi
  # shellcheck disable=SC2086
  run $bench_bin \
    --base-url "$base_url" \
    --model "$model" \
    --tokenizer "$tokenizer" \
    --pp "$pp" \
    --tg "$tg" \
    --depth "${depths[@]}" \
    --concurrency 1 \
    --runs "$runs" \
    --format md
fi

echo "tuning table: $out"
