#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=96:00:00
#SBATCH --mem=50G
module load singularity
singularity exec docker_image.sif python truckie/main.py --comp deigo --arg_name e_2 --alpha None --target_entropy -1 --agents $agents_per_job --previous_agents $previous_agents
