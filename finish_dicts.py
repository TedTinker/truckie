#------------------
# finish_dicts.py merges each agent's value_dict_###.pickle and min_max_dict_###.pickle
# into one value_dict.pickle and one min_max_dict.pickle per arg_name, then deletes
# the per-agent files it merged.
#
#   --temp True   merge the _temp_ files written mid-training (GUI pings) instead.
#
# Safe to rerun: merged output is written to a .partial file and renamed into place,
# so a job killed partway leaves the per-agent files untouched and the old merged
# file intact. Per-agent files are only deleted after the new merged file exists.
#------------------

import gc
import os
import pickle
import re

from utils import duration, args, print, save_file

print('name:\n{}'.format(args.arg_name))

TAG = 'temp_' if args.temp else ''
PATTERN = re.compile(rf'^(value_dict|min_max_dict)_{TAG}(\d+)\.pickle$')     # any number of digits
TEMP_PATTERN = re.compile(r'^(value_dict|min_max_dict)_temp_(\d+)\.pickle$')
SINGLE_KEYS = ['args', 'arg_title', 'arg_name']         # the same for every agent; kept once

os.chdir(save_file)
complete_order = args.arg_title[3:-3].split('+')
folders = [o for o in complete_order if o not in ['empty_space', 'break']]



def agent_files(folder, pattern):
    """(kind, index, filename) for matching files, in agent order. Sorted by the
    number, not the text, so agent 1000 comes after agent 999."""
    found = []
    for file in os.listdir(folder):
        match = pattern.match(file)
        if match:
            found.append((match.group(1), int(match.group(2)), file))
    return sorted(found, key = lambda f: (f[0], f[1]))


def merge_into(merged, saved):
    for key, value in saved.items():
        if key in SINGLE_KEYS:
            merged[key] = value
        else:
            merged.setdefault(key, []).append(value)


def truncate_lists_in_dict(d):
    """Where every agent has a list, cut them all to the shortest length."""
    for key, value in d.items():
        if isinstance(value, list) and value and all(isinstance(item, list) for item in value):
            lengths = [len(lst) for lst in value if lst]
            min_length = min(lengths) if lengths else 0
            d[key] = [lst[:min_length] for lst in value]
    return d


def reduce_min_max(min_max_dict):
    """Each key holds one (min, max) per agent; keep the overall (min, max)."""
    for key, per_agent in min_max_dict.items():
        if key in SINGLE_KEYS:
            continue
        minimum = maximum = None
        for min_max in per_agent:
            if not isinstance(min_max, (tuple, list)) or len(min_max) != 2:
                continue
            if any(item is None or isinstance(item, str) for item in min_max):
                continue
            if minimum is None or min_max[0] < minimum:
                minimum = min_max[0]
            if maximum is None or min_max[1] > maximum:
                maximum = min_max[1]
        min_max_dict[key] = (minimum, maximum)
    return min_max_dict


def dump_atomic(obj, path):
    """Write to path.partial, then rename. A killed job never leaves a half-written file."""
    partial = path + '.partial'
    with open(partial, 'wb') as handle:
        pickle.dump(obj, handle, protocol = pickle.HIGHEST_PROTOCOL)
    os.replace(partial, path)



for folder in folders:
    if not os.path.isdir(folder):
        print(f'No folder {folder}, skipping.')
        continue

    files = agent_files(folder, PATTERN)
    if not files:
        print(f'No matching files in {folder}, skipping.')
        continue
    print(f'{folder}: merging {len(files)} files.')

    # One file in memory at a time; each is merged, then dropped.
    value_dict, min_max_dict = {}, {}
    for kind, index, file in files:
        with open(os.path.join(folder, file), 'rb') as handle:
            saved = pickle.load(handle)
        merge_into(value_dict if kind == 'value_dict' else min_max_dict, saved)
        del saved

    # Older agents send these as one dict each; merge them into one dict, if present.
    for key in ['episode_dicts', 'agent_lists']:
        if key in value_dict:
            combined = {}
            for d_item in value_dict[key]:
                combined.update(d_item)
            value_dict[key] = combined

    value_dict = truncate_lists_in_dict(value_dict)
    min_max_dict = reduce_min_max(min_max_dict)

    dump_atomic(min_max_dict, os.path.join(folder, 'min_max_dict.pickle'))
    dump_atomic(value_dict, os.path.join(folder, 'value_dict.pickle'))
    del value_dict, min_max_dict
    gc.collect()

    # Only now, with the merged files safely written, delete what went into them.
    for _, _, file in files:
        os.remove(os.path.join(folder, file))
    removed = len(files)

    # A final merge makes any mid-training snapshots obsolete.
    if not args.temp:
        for _, _, file in agent_files(folder, TEMP_PATTERN):
            os.remove(os.path.join(folder, file))
            removed += 1

    # Leftovers from a run that was killed while writing.
    for file in os.listdir(folder):
        if file.endswith('.partial'):
            os.remove(os.path.join(folder, file))

    print(f'Finished {folder}: removed {removed} per-agent files.')

print('\nDuration: {}. Done!\n'.format(duration()))