#!/bin/bash -l

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name_of_folder="$(basename "$project_root")"

read_config() {
    python3 -c "import json, sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" \
        "$project_root/config.json" "$1"
}

sif_file=$(read_config sif_file)
max_agents=$(read_config max_agents_per_job)

eval $1
eval $2
eval $3

real_arg_list=()

for arg in ${arg_list[*]}
do
    temp_file=$(mktemp)
    singularity exec ${sif_file}.sif python -c "from ${name_of_folder}.bash.slurmcraft import all_like_this; result = all_like_this('$arg'); print(result, file=open('${temp_file}', 'w'))"
    returned_value=$(cat ${temp_file})
    #real_arg_list=$(echo "${real_arg_list}${returned_value}" | jq -s 'add')
    real_arg_list=$(python -c "import json; a = json.loads('""$real_arg_list""') if '""$real_arg_list""' else []; b = json.loads('""$returned_value""') if '""$returned_value""' else []; print(json.dumps(a+b))")
    rm ${temp_file}
done

arg_list=$real_arg_list
singularity exec ${sif_file}.sif python ${name_of_folder}/bash/slurmcraft.py --comp ${comp} --agents ${agents} --arg_list "${arg_list}"
arg_list=$(echo "${arg_list}" | tr -d '[]"' | sed 's/,/, /g')
wait

echo
if [ $agents -eq 0 ]; then
    dict_jid=$(sbatch --partition=compute --wrap="sleep .1" | awk '{print $4}')
    echo "No training"
else
    jid_list=()
    for arg in ${arg_list//, / }
    do
        previous_agents=0 
        if [ $arg == "break" ]
        then
            :
        elif [ $arg == "empty_space" ]
        then
            :
        elif [ ${agents} -gt ${max_agents} ]
        then
            num_jobs=$(( ${agents} / ${max_agents} )) 
            remainder=$(( ${agents} % ${max_agents} ))
            if [ $remainder -gt 0 ]
            then
                num_jobs=$(( num_jobs + 1 ))
            else 
                remainder=${max_agents}
            fi
            for (( i=1; i<=${num_jobs}; i++ ))
            do
                if [ $i -eq ${num_jobs} ]
                then
                    agents_per_job=$(( remainder ))
                else
                    agents_per_job=$(( ${max_agents} ))
                fi
                jid=$(sbatch --export=agents_per_job=${agents_per_job},previous_agents=${previous_agents} ${name_of_folder}/bash/main_${arg}.slurm | awk '{print $4}')
                echo "$jid : $arg ($i)"
                jid_list+=($jid)
                previous_agents=$(( previous_agents + agents_per_job ))
            done
        else
            jid=$(sbatch --export=agents_per_job=${agents},previous_agents=0 ${name_of_folder}/bash/main_${arg}.slurm | awk '{print $4}')
            echo "$jid : $arg"
            jid_list+=($jid)
        fi
    done
    job_ids=$(echo ${jid_list[@]} | tr ' ' ':')  
    dict_jid=$(sbatch --dependency=afterok:${job_ids} ${name_of_folder}/bash/finish_dicts.slurm | awk '{print $4}')
    echo
    echo "$dict_jid : finishing dictionaries"
fi