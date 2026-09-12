#!/bin/bash
# v_macro re-fit with the open detector: every released SB2 + 10,000 unflagged
# searched stars. Same fitting layer and net as job 50445224, seeded per star,
# saving the fitted broadening and full label vectors. 16 x 32 CPUs x 55 min
# caps the worst case at 469 CPU-h; expected ~130 CPU-h.
#SBATCH --job-name=a4b_vmacro
#SBATCH --account=PAS2823
#SBATCH --array=0-15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:55:00
#SBATCH --output=logs/a4b_vmacro_%A_%a.out
set -e
module load python/3.12 2>/dev/null || true
source "$HOME/a4b_venv/bin/activate"
cd "$HOME/agent4binary"
export OMP_NUM_THREADS=1
export AB_MLP_NET=models/payne_dr19_sfbig_holdA.pt
ID=$SLURM_ARRAY_TASK_ID
python src/run_census_stage1.py --detector binspec_mlp --no-download \
    --manifest resources/census_vmacro_shards/manifest_shard_${ID}.csv \
    --raw-dir /fs/scratch/PAS2823/a4b_census272k_raw \
    --out resources/census/vmacro_refit_${ID}.csv \
    --workers 32 --checkpoint-every 200
echo "VMACRO_SHARD_${ID}_DONE"
