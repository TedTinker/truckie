#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=30
#SBATCH --time=96:00:00
#SBATCH --mem=50G
module load singularity
singularity exec docker_image.sif python truckie/main.py --comp deigo --arg_name ehv_layers_8 --alpha None --target_entropy -1 --curiosity hidden_state --hidden_state_eta_vision 0.01 --hidden_state_eta_touch 0.1 --hidden_state_eta_prop 0.03 --hidden_state_eta_feedback_voice 0 --shared_voice_encoder True --layers 3 --agents $agents_per_job --previous_agents $previous_agents
