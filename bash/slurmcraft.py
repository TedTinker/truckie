#%%
import sys, json, argparse
from copy import deepcopy
from pathlib import Path

BASH_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASH_DIR.parent))
from config import FOLDER_NAME, SIF_FILE, MAX_AGENTS_PER_JOB, CLUSTERS

parser = argparse.ArgumentParser()
parser.add_argument('--comp',     type=str, default='deigo')
parser.add_argument('--agents',   type=int, default=10)
parser.add_argument('--arg_list', type=str, default=[])
try:    args = parser.parse_args()
except: args, _ = parser.parse_known_args()

if type(args.arg_list) != list:
    args.arg_list = json.loads(args.arg_list)
combined = '___{}___'.format('+'.join(args.arg_list))



# Given sets of arguments, return all possible combinations.
def expand_args(name, args):
    combos = [{}]
    complex = False
    for key, value in args.items():
        if type(value) != list:
            for combo in combos:
                combo[key] = value
        else: 
            complex = True
            if value[0] == 'num_min_max': 
                num, min_val, max_val = value[1]
                num = int(num)
                min_val = float(min_val)
                max_val = float(max_val)
                value = [min_val + i*((max_val - min_val) / (num - 1)) for i in range(num)]
            new_combos = []
            for v in value:
                temp_combos = deepcopy(combos)
                for combo in temp_combos: 
                    combo[key] = v 
                    new_combos.append(combo)   
            combos = new_combos  
    if complex and name[-1] != '_': 
        name += '_'
    return(name, combos)

def convert_list(input_list):
    converted = ['\\[' + ','.join(map(str, sub_list)) + '\\]' for sub_list in input_list]
    return(converted)



# Default set of arguments.
slurm_dict = {'d' : {}}



# Add these arguments to slurm_dict.
def add_this(name, args):
    keys, values = [], []
    for key, value in slurm_dict.items(): keys.append(key) ; values.append(value)
    for key, value in zip(keys, values):  
        if key == 'd': 
            key = ''
        between = '' if key == '' or len(name) == 1 else '_'
        new_key = key + between + name 
        new_value = deepcopy(value)
        for arg_name, arg in args.items():
            if type(arg) != list: 
                new_value[arg_name] = arg
            elif type(arg[0]) != list: 
                new_value[arg_name] = arg
            else:
                for condition in arg:
                    for if_arg_name, if_arg in condition[0].items():
                        if if_arg_name in value and value[if_arg_name] == if_arg:
                            new_value[arg_name] = condition[1]
        slurm_dict[new_key] = new_value



# Agents with entropy.
add_this('e',   {
    'alpha' : 'None', 
    'action_cost' : [0, .1],
    'target_entropy' : [-2, -1, -.5]})    



# Agents with curiosity.
add_this('h',   {
    'curiosity' : 'hidden_state',
    'hidden_state_eta_vision' : .1, 
    'hidden_state_eta_touch' : 1,
    'hidden_state_eta_prop' : .3,
    'hidden_state_eta_feedback_voice' : [0, .3]})    




add_this('quick',   {
    'epochs' : 1000,
    'epochs_per_agent_save' : 250})    



new_slurm_dict = {}
for key, value in slurm_dict.items():
    key, combos = expand_args(key, value)
    if len(combos) == 1: 
        new_slurm_dict[key] = combos[0] 
    else:
        for i, combo in enumerate(combos): new_slurm_dict[key + str(i+1)] = combo
        
slurm_dict = new_slurm_dict

def get_args(name):
    s = '' 
    for key, value in slurm_dict[name].items(): s += '--{} {} '.format(key, value)
    return(s)

def all_like_this(this): 
    if this in ['break', 'empty_space']: 
        result = [this]
    elif this[-1] != '_':                
        result = [this]
    else: 
        result = [key for key in slurm_dict.keys() if key.startswith(this) and key[len(this):].isdigit()]
    return(json.dumps(result))



if __name__ == '__main__':
    for arg_name, arg_list in slurm_dict.items():
        print(f'{arg_name}: \t {arg_list}')

            

max_cpus = min(args.agents, MAX_AGENTS_PER_JOB)

def slurm_header(comp, cpus=1):
    c = CLUSTERS[comp]
    lines = ['#!/bin/bash -l',
             f'#SBATCH --partition={c["partition"]}',
             '#SBATCH --nodes=1',
             '#SBATCH --ntasks=1',
             f'#SBATCH --cpus-per-task={cpus}',
             f'#SBATCH --time={c["time"]}',
             f'#SBATCH --mem={c["mem"]}']
    if c['gres']:
        lines.append(f'#SBATCH --gres={c["gres"]}')
    lines.append(c['module'])
    return '\n'.join(lines) + '\n'



if __name__ == '__main__' and args.arg_list != []:
    run = f'singularity exec{CLUSTERS[args.comp]["nv"]} {SIF_FILE}.sif python {FOLDER_NAME}'

    for name in args.arg_list:
        if name in ['break', 'empty_space']:
            continue
        (BASH_DIR / f'main_{name}.slurm').write_text(
            slurm_header(args.comp, cpus=max_cpus) +
            f'{run}/main.py --comp {args.comp} --arg_name {name} {get_args(name)}'
            f'--agents $agents_per_job --previous_agents $previous_agents\n')

    (BASH_DIR / 'finish_dicts.slurm').write_text(
        slurm_header(args.comp) +
        f'{run}/finish_dicts.py --comp {args.comp} --arg_title {combined} '
        f'--arg_name finishing_dictionaries\n')

# %%

