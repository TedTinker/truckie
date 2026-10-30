#%%

import os
import gzip
import pickle
import torch
import random
import numpy as np
from multiprocessing import Process, Queue, set_start_method
from time import sleep
from math import floor

from utils import args, folder, duration, estimate_total_duration, print, cpu_memory_usage, duration, print_duration, wait_for_button_press
from agent import Agent, latest_agent_file

print('\nname:\n{}'.format(args.arg_name))
print('\nagents: {}. previous_agents: {}.'.format(args.agents, args.previous_agents))


def train(q, i):
    """Train one agent (i) and send progress updates to queue q."""
    torch.set_num_threads(1)            # one core per agent; otherwise every agent
    torch.set_num_interop_threads(1)    # tries to use all of them at once
    seed = args.init_seed + i
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed(int(seed))

    if str(args.device) != 'cpu':
        num_gpus = torch.cuda.device_count()
        gpu_id = i % num_gpus
        args.device = torch.device(f'cuda:{gpu_id}')

    num_cores = os.cpu_count()
    cpu_id = i % num_cores
    args.cpu = cpu_id

    print(f'\nagent {i}: cpu {cpu_id}\n')

    load_path = latest_agent_file(i) if args.load_agents else None
    if load_path is not None:
        print("LOADING", i, load_path)
        with gzip.open(load_path, 'rb') as handle:
            agent = pickle.load(handle)     # Continues from its saved epoch, with an empty buffer.
        agent.resume(args)
    else:
        if args.load_agents:
            print(f"No saved agent {i} in {folder}/agents; starting a new one.")
        agent = Agent(args=args, i=i)

    agent.training(q)


if __name__ == '__main__':
    """Main entry point for multi-agent training."""
    set_start_method('spawn')  # Required for multiprocessing
    queue = Queue()
    processes = []

    for worker_id in range(1 + args.previous_agents, 1 + args.agents + args.previous_agents):
        process = Process(target=train, args=(queue, worker_id))
        processes.append(process)
        process.start()

    # Progress tracking. Agents may send progress as a float or a string; store floats.
    agent_ids = range(1 + args.previous_agents, 1 + args.agents + args.previous_agents)
    progress_dict      = {i: 0.0  for i in agent_ids}
    prev_progress_dict = {i: None for i in agent_ids}

    while any(process.is_alive() for process in processes) or not queue.empty():
        while not queue.empty():
            worker_id, progress = queue.get()
            progress_dict[worker_id] = float(progress)

        # If there's been any progress update, print the new state.
        if progress_dict != prev_progress_dict:
            prev_progress_dict = progress_dict.copy()

            values = sorted(progress_dict.values())
            so_far = duration()
            estimated_total = estimate_total_duration(values[0])
            to_do = '?:??:??' if estimated_total == '?:??:??' else estimated_total - so_far

            values_display = []
            hundreds = 0
            for value in values:
                val_str = str(floor(100 * value)).ljust(3, ' ')
                if val_str.strip() == '100':
                    hundreds += 1
                else:
                    values_display.append(val_str)

            bar = ' '.join(values_display) + ' ##' + ' 100' * hundreds
            bar = f'{so_far} ({to_do} left):\t' + bar.rstrip() + '.'
            print(bar)

        sleep(15)

    for process in processes:
        process.join()

    print('\nDuration: {}. Done!'.format(duration()))