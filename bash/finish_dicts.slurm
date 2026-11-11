#!/bin/bash -l
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=96:00:00
#SBATCH --mem=50G
module load singularity
singularity exec docker_image.sif python truckie/finish_dicts.py --comp deigo --arg_title ___v+ev_1+ev_2+ev_3+ehv_1+ehv_2+ehv_3+ehv_4+ehv_5+ehv_6+v_layers_1+v_layers_2+ev_layers_1+ev_layers_2+ev_layers_3+ev_layers_4+ev_layers_5+ev_layers_6+ehv_layers_1+ehv_layers_2+ehv_layers_3+ehv_layers_4+ehv_layers_5+ehv_layers_6+ehv_layers_7+ehv_layers_8+ehv_layers_9+ehv_layers_10+ehv_layers_11+ehv_layers_12___ --arg_name finishing_dictionaries
