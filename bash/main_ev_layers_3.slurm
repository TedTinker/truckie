#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=96:00:00
#SBATCH --mem=50G
module load singularity
singularity exec docker_image.sif python truckie/main.py --comp deigo --arg_name ev_layers_3 --alpha None --target_entropy -0.5 --shared_voice_encoder True --layers 2 --agents $agents_per_job --previous_agents $previous_agents
