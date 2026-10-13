#!/bin/bash
#SBATCH --job-name=disc-grid
#SBATCH --account=def-mbalogh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-239%120
#SBATCH --output=/project/def-mbalogh/jdufr/output_gamma1.0_t1.0Myr/logs/%A_%a.out
#SBATCH --error=/project/def-mbalogh/jdufr/output_gamma1.0_t1.0Myr/logs/%A_%a.err

set -eo pipefail
module load python/3.11
source "$HOME/discevolution-env/bin/activate"
set -u
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export MPLBACKEND=Agg
export MPLCONFIGDIR="${SLURM_TMPDIR:?}/matplotlib"
mkdir -p "$MPLCONFIGDIR"

config=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" manifest.txt)
if [[ -z "$config" || ! -f "$config" ]]; then
    echo "Missing config for task $SLURM_ARRAY_TASK_ID" >&2
    exit 1
fi
tag=$(basename "$config" .json)
# Every submission gets a separate directory, including retries.
output_root="/project/def-mbalogh/jdufr/output_gamma1.0_t1.0Myr"
output="${output_root}/${tag}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$output"
cp "$config" "$output/config.json"
echo "Config: $config"
echo "Output: $output"
# Slurm controls wall time; disable the driver's predictive ETA cutoff.
srun python -u run_model_testNoSurrogate.py --config "$config" \
    --output_dir "$output" --eta_abort_hours inf
