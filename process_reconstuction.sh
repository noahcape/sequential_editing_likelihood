#!/bin/bash
#SBATCH --job-name=apc_project_process_recon            # create a short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=5G            # memory per cpu-core (4G is default)
#SBATCH --time=05:00:00             # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin,end,fail  # receive email notifications
#SBATCH --mail-user=nc4935@princeton.edu

module purge
module load anaconda3/2025.6

PROJ_DIR="/home/nc4935/apc523/sequential_editing_likelihood"
PROCESS_RECON="${PROJ_DIR}/py/process_reconstruction.py"

#activate the virtual env

echo "Processing larger pre-reconstructed instance ./simulated_data/0"
python $PROCESS_RECON

