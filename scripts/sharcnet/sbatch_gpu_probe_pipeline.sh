#!/usr/bin/env bash
# =============================================================================
# Slurm batch script: full prerequisite pipeline + downstream probe grid (GPU).
# Use on Alliance (e.g. Narval), SHARCNET, or any Slurm+CUDA site after editing #SBATCH lines.
#
# Trains all four conditions × three seeds in one process via run_probe_grid.py
# (12 runs total: supervised, moco, swav × 50-epoch probe; random_init × 100 epochs).
#
# Contract (orchestration only in this file):
# - This shell script does not embed training logic; it calls CLIs that use src/cv/train/trainer.py.
# - Only invokes existing scripts under scripts/ via `uv run python`.
#
# Submit from the repo (or any directory under it); Slurm sets SLURM_SUBMIT_DIR to that cwd.
# The batch file is often *copied* to /localscratch/spool/slurmd/jobNNN — do not rely on the
# script path there. We locate pyproject.toml by walking up from SLURM_SUBMIT_DIR first.
# Optional: export REPO_ROOT=/abs/path/to/pretraining-saliency-downstream-classifiers before sbatch.
#
# Load modules or activate the environment before training if your site requires it
# (example, adjust to your cluster docs):
#   - module load StdEnv/2023 cuda cudnn python/3.11
#
# Optional environment overrides (export before sbatch, or use #SBATCH --export=ALL):
#  - STRICT_CANONICAL=1 -> match docs/requirements.md Stage 4 (num_workers=0, no pin_memory)
#  - NUM_WORKERS=8 -> dataloader workers when STRICT_CANONICAL is unset (default 8)
#  - NO_STRICT_REPRO=1 -> pass --no-strict-repro (faster: cudnn.benchmark + CUDA AMP; less reproducible)
#  - NO_AMP=1 -> pass --no-amp (fp32 on GPU; only matters if NO_STRICT_REPRO=1)
#  - SKIP_PREP=1->  skip prepare_encoders + make_splits (artifacts already exist)
#  - RUN_SALIENCY=1-> after training, run generate_explanations + qc_explanations
#  - EXPLAIN_DEVICE=cuda  -> device for saliency stage (default cuda if RUN_SALIENCY=1)
# Two GPUs: this script runs a single Python process (run_probe_grid.py), which uses one GPU.
# =============================================================================

#SBATCH --job-name=probe-grid-gpu
#SBATCH --time=04:00:00
# Time: CPU runs can be many hours; the CUDA grid alone is usually far shorter on an A100.
# Raise this if you add RUN_SALIENCY=1 or hit time limits; Slurm kills jobs that exceed --time.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --account=def-<account_name>
#SBATCH --mail-type=ALL
#SBATCH --mail-user=<user_name>@uwo.ca
##SBATCH --partition=REPLACE_WITH_YOUR_GPU_PARTITION

set -euo pipefail

# Find repo root (directory containing pyproject.toml). Slurm runs a *copy* of this script under
# /localscratch/... so BASH_SOURCE is useless for locating the clone on $HOME or $PROJECT.
_walk_up_to_repo() {
  local d
  d="$(cd "$1" && pwd)" || return 1
  local _i
  for _i in {1..16}; do
    if [[ -f "${d}/pyproject.toml" ]]; then
      echo "${d}"
      return 0
    fi
    if [[ "${d}" == "/" ]]; then
      return 1
    fi
    d="$(dirname "${d}")"
  done
  return 1
}

REPO_ROOT="${REPO_ROOT:-}"
if [[ -n "${REPO_ROOT}" ]] && [[ -f "${REPO_ROOT}/pyproject.toml" ]]; then
  REPO_ROOT="$(cd "${REPO_ROOT}" && pwd)"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  REPO_ROOT="$(_walk_up_to_repo "${SLURM_SUBMIT_DIR}")" || REPO_ROOT=""
fi
if [[ -z "${REPO_ROOT}" ]]; then
  _fallback_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  REPO_ROOT="$(_walk_up_to_repo "${_fallback_dir}")" || REPO_ROOT=""
fi
if [[ -z "${REPO_ROOT}" ]] || [[ ! -f "${REPO_ROOT}/pyproject.toml" ]]; then
  echo "ERROR: pyproject.toml not found." >&2
  echo "Fix: run sbatch from inside the repo (e.g. cd .../pretraining-saliency-downstream-classifiers && sbatch scripts/sharcnet/sbatch_gpu_probe_pipeline.sh)" >&2
  echo "Or:  export REPO_ROOT=/abs/path/to/pretraining-saliency-downstream-classifiers before sbatch." >&2
  echo "SLURM_SUBMIT_DIR=${SLURM_SUBMIT_DIR:-<unset>}" >&2
  exit 1
fi
cd "${REPO_ROOT}"

echo "Host: $(hostname)"
echo "Repo: ${REPO_ROOT}"
echo "Date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
if command -v nvidia-smi &>/dev/null; then
  nvidia-smi || true
fi

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not on PATH. Load your environment or install uv on the cluster." >&2
  exit 1
fi

UV_RUN=(uv run python)

# Stage 4 flags: canonical vs practical GPU (see docs/requirements.md)
GRID_HW_FLAGS=()
if [[ "${STRICT_CANONICAL:-0}" == "1" ]]; then
  GRID_HW_FLAGS=(--num-workers 0 --no-pin-memory)
  echo "STRICT_CANONICAL=1: using --num-workers 0 --no-pin-memory (matches docs/requirements.md)"
else
  GRID_HW_FLAGS=(--num-workers "${NUM_WORKERS:-8}")
  echo "Using --num-workers ${GRID_HW_FLAGS[1]} (default pin_memory=True). Set STRICT_CANONICAL=1 for notebook-exact flags."
fi

GRID_REPRO_FLAGS=()
if [[ "${NO_STRICT_REPRO:-0}" == "1" ]]; then
  GRID_REPRO_FLAGS+=(--no-strict-repro)
  echo "NO_STRICT_REPRO=1: --no-strict-repro (cudnn.benchmark + AMP allowed for speed)"
fi
if [[ "${NO_AMP:-0}" == "1" ]]; then
  GRID_REPRO_FLAGS+=(--no-amp)
  echo "NO_AMP=1: --no-amp (fp32 on GPU)"
fi

if [[ "${SKIP_PREP:-0}" != "1" ]]; then
  echo "Stage 1: prepare encoders (CPU)"
  "${UV_RUN[@]}" scripts/prepare_encoders.py \
    --conditions supervised moco swav \
    --device cpu

  echo "Stage 2: fixed splits"
  "${UV_RUN[@]}" scripts/make_splits.py \
    --split-seed 42 \
    --val-ratio 0.2 \
    --download
else
  echo "SKIP_PREP=1: skipping prepare_encoders and make_splits"
fi

echo "Stage 4: probe grid (CUDA, explicit --device cuda)"
"${UV_RUN[@]}" scripts/run_probe_grid.py \
  --conditions supervised moco swav random_init \
  --seeds 0 1 2 \
  --device cuda \
  --download \
  --allow-remote-download \
  "${GRID_HW_FLAGS[@]}" \
  "${GRID_REPRO_FLAGS[@]}"

if [[ "${RUN_SALIENCY:-0}" == "1" ]]; then
  _explain_dev="${EXPLAIN_DEVICE:-cuda}"
  echo "==> Stage 5: saliency (${_explain_dev})"
  "${UV_RUN[@]}" scripts/generate_explanations.py \
    --conditions supervised moco swav random_init \
    --seeds 0 1 2 \
    --methods gradcam gradcampp occlusion \
    --batch-size 8 \
    --device "${_explain_dev}"

  echo "==> Stage 6: saliency QC"
  "${UV_RUN[@]}" scripts/qc_explanations.py \
    --conditions supervised moco swav random_init \
    --seeds 0 1 2 \
    --methods gradcam gradcampp occlusion
else
  echo "RUN_SALIENCY not set: skipping generate_explanations and qc_explanations"
fi

echo "Done."
