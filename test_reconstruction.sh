#!/bin/bash
#SBATCH --job-name=apc_project_recon            # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --output=test_recon.out
#SBATCH --error=test_recon.err
#SBATCH --cpus-per-task=4           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=5G            # memory per cpu-core (4G is default)
#SBATCH --time=05:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=nc4935@princeton.edu

module purge
module load anaconda3/2025.6

source .venv/bin/activate

export JAX_LOG_COMPILES=1

PROJ_DIR="/home/nc4935/apc523/sequential_editing_likelihood"
TREE_SEARCH="${PROJ_DIR}/py/run_tree_search_instance.py"

# run on small_simulated_data/0
srun python $TREE_SEARCH \
    0 \
    --simulation-dir "${PROJ_DIR}/small_simulated_data"
