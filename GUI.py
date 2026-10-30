#%%

import os
import re
import subprocess
import threading
import tkinter as tk
from tkinter import ttk
from collections import defaultdict
from natsort import natsorted
import shutil

# This file is only for viewing data for robots with various arguments mid-training.

name_of_folder = "curious_maze"

# It may be important to observe only a few robot's data, in order to conserve memory.
files_to_create = [i for i in range(1, 11)]


def parse_slurm_files():
    """
    Parse slurm files and extract mappings from arg_name to filenames.
    """
    arg_name_to_slurm_files = defaultdict(list)
    slurm_files = [f for f in os.listdir('.') if f.startswith('slurm-') and f.endswith('.out')]
    for slurm_file in slurm_files:
        with open(slurm_file, 'r') as f:
            lines = f.readlines()
        arg_name = None
        for i, line in enumerate(lines):
            if line.strip().startswith('arg_name:'):
                j = i + 1
                while j < len(lines) and lines[j].startswith('\t'):
                    if 'This time:' in lines[j]:
                        match = re.search(r'This time:\s*(.*)', lines[j])
                        if match:
                            arg_name = match.group(1).strip()
                            break
                    j += 1
                if arg_name:
                    break
        if arg_name:
            arg_name_to_slurm_files[arg_name].append(slurm_file)
    return arg_name_to_slurm_files


class ArgNameData:
    """
    Tracks metadata for a single arg_name, including slurm files and status flags.
    """
    def __init__(self, arg_name):
        self.arg_name = arg_name
        self.slurm_files = []
        self.files_to_create = files_to_create
        self.files_created = False
        self.singularity_run = False
        self.files_listbox = None

    def update_slurm_files(self, slurm_files):
        """
        Update the slurm file list for this argument name.
        """
        self.slurm_files = slurm_files
        self.files_to_create = files_to_create


def create_files_for_arg_name_plot_dict(arg_name_data):
    """
    Create ping files in folder to trigger plot dict saving for each agent.
    """
    arg_name = arg_name_data.arg_name
    for i in arg_name_data.files_to_create:
        agent_name = f'{arg_name}_{i}'
        file_path = os.path.join(name_of_folder, 'saved_deigo', f'{arg_name}', f'save_{arg_name}_{i}_plot_dict')
        open(file_path, 'w').close()
    arg_name_data.files_created = True
    arg_name_data.singularity_run = False


def create_files_for_arg_name_save_agent(arg_name_data):
    """
    Create ping files in folder to trigger agent saving for each agent.
    """
    arg_name = arg_name_data.arg_name
    for i in arg_name_data.files_to_create:
        agent_name = f'{arg_name}_{i}'
        file_path = os.path.join(name_of_folder, 'saved_deigo', f'{arg_name}', f'save_{arg_name}_{i}_agent')
        open(file_path, 'w').close()
        
def create_files_for_arg_name_save_agent_single(arg_name_data, index):
    """
    Create a ping file to trigger agent saving for a single agent index.
    """
    arg_name = arg_name_data.arg_name
    file_path = os.path.join(name_of_folder, 'saved_deigo', f'{arg_name}', f'save_{arg_name}_{index}_agent')
    open(file_path, 'w').close()


def check_files_for_arg_name_plot_dict(arg_name_data):
    """
    Check if required files exist, and trigger merging script if missing.
    """
    if not arg_name_data.files_created:
        return
    if arg_name_data.singularity_run:
        return
    arg_name = arg_name_data.arg_name
    dir_path = os.path.join(name_of_folder, 'saved_deigo', arg_name)
    existing_files = [f for f in os.listdir(dir_path) if f.startswith(f'{arg_name}_')]
    if not existing_files:
        run_python_command(arg_name_data)


def run_python_command(arg_data):
    """
    Run the dictionary-merging script for the specified arg_name in a background thread.
    """
    def run_command():
        arg_name = arg_data.arg_name
        cmd = [
            'python', f'{name_of_folder}/finish_dicts.py',
            '--comp', 'deigo',
            '--arg_title', f'___{arg_name}___',
            '--arg_name', 'finishing_dictionaries',
            '--temp', 'True'
        ]
        subprocess.run(cmd)
    arg_data.singularity_run = True
    threading.Thread(target=run_command).start()


class GUIApp:
    """
    GUI application for managing and monitoring robots by argument name.
    """
    def __init__(self, root):
        self.root = root
        self.root.title('Arg Name Monitor')
        self.arg_name_data_dict = {}
        self.arg_name_frames = {}
        self.max_columns = 9
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        self.update_interval = 2000
        self.build_ui()
        self.update_data()

    def build_ui(self):
        """
        Build the GUI layout and widget components.
        """
        self.canvas = tk.Canvas(self.main_frame)
        self.scrollbar = tk.Scrollbar(self.main_frame, orient='vertical', command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.scrollbar.pack(side='right', fill='y')

        self.scrollable_frame = tk.Frame(self.canvas)
        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        )
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')

        all_buttons_frame = tk.Frame(self.main_frame)
        all_buttons_frame.pack(side='bottom', pady=5)

        all_button = tk.Button(all_buttons_frame, text='Create Files for All', command=self.create_files_for_all)
        all_button.pack(side='left', padx=5)
        
        save_agent_all_button = tk.Button(all_buttons_frame, text='Save Agent for All', command=self.save_agent_for_all)
        save_agent_all_button.pack(side='left', padx=5)

        delete_all_button = tk.Button(all_buttons_frame, text='Delete Files for All', command=self.delete_files_for_all)
        delete_all_button.pack(side='left', padx=5)

    def create_files_for_all(self):
        """
        Create plot dict ping files for all arg_names currently loaded.
        """
        for arg_data in self.arg_name_data_dict.values():
            self.create_files(arg_data)

    def delete_files_for_all(self):
        """
        Delete files for all arg_names currently loaded.
        """
        for arg_data in self.arg_name_data_dict.values():
            self.delete_files(arg_data)

    def save_agent_for_all(self):
        """
        Trigger agent saving for all arg_names currently loaded.
        """
        for arg_data in self.arg_name_data_dict.values():
            self.save_agent(arg_data)

    def update_data(self):
        """
        Refresh the list of arg_names and update their corresponding frames.
        """
        arg_name_to_slurm_files = parse_slurm_files()
        for arg_name in natsorted(arg_name_to_slurm_files.keys()):
            slurm_files = arg_name_to_slurm_files[arg_name]
            if arg_name not in self.arg_name_data_dict:
                arg_data = ArgNameData(arg_name)
                arg_data.update_slurm_files(slurm_files)
                self.arg_name_data_dict[arg_name] = arg_data
                self.add_arg_name_frame(arg_data)
            else:
                arg_data = self.arg_name_data_dict[arg_name]
                arg_data.update_slurm_files(slurm_files)

        existing_arg_names = set(arg_name_to_slurm_files.keys())
        for arg_name in list(self.arg_name_data_dict.keys()):
            if arg_name not in existing_arg_names:
                self.remove_arg_name_frame(arg_name)

        for arg_name, arg_data in self.arg_name_data_dict.items():
            self.update_arg_name_frame(arg_data)

        self.reposition_arg_frames()
        self.root.after(self.update_interval, self.update_data)

    def reposition_arg_frames(self):
        """
        Rearrange argument name frames in a grid layout.
        """
        sorted_keys = natsorted(self.arg_name_data_dict.keys())
        for idx, arg_name in enumerate(sorted_keys):
            frame = self.arg_name_frames[arg_name]
            row = idx // self.max_columns
            col = idx % self.max_columns
            frame.grid_configure(row=row, column=col, padx=5, pady=5, sticky='n')

    def add_arg_name_frame(self, arg_data):
        """
        Create and display a frame for the given arg_name.
        """
        frame = tk.Frame(self.scrollable_frame, bd=2, relief=tk.GROOVE)

        label = tk.Label(frame, text=f'arg_name: {arg_data.arg_name}')
        label.pack(side=tk.TOP, anchor='w')

        create_button = tk.Button(frame, text='Create Files', command=lambda arg_data=arg_data: self.create_files(arg_data))
        create_button.pack(side=tk.TOP, anchor='w')
        
        save_agent_frame = tk.Frame(frame)
        save_agent_frame.pack(side=tk.TOP, anchor='w', fill=tk.X)
        for i in arg_data.files_to_create:
            btn = tk.Button(
                save_agent_frame,
                text=str(i),
                command=lambda idx=i, ad=arg_data: create_files_for_arg_name_save_agent_single(ad, idx)
            )
            btn.pack(side=tk.LEFT)

        delete_button = tk.Button(frame, text='Delete Files', command=lambda arg_data=arg_data: self.delete_files(arg_data))
        delete_button.pack(side=tk.TOP, anchor='w')

        files_label = tk.Label(frame, text='Files in folder:')
        files_label.pack(side=tk.TOP, anchor='w')

        files_listbox = tk.Listbox(frame)
        files_listbox.pack(side=tk.TOP, fill=tk.X, expand=True)
        arg_data.files_listbox = files_listbox

        frame.grid(row=0, column=0, padx=5, pady=5, sticky='n')
        self.arg_name_frames[arg_data.arg_name] = frame

    def remove_arg_name_frame(self, arg_name):
        """
        Remove the frame associated with the given arg_name.
        """
        frame = self.arg_name_frames.pop(arg_name)
        frame.destroy()
        self.arg_name_data_dict.pop(arg_name)

    def update_arg_name_frame(self, arg_data):
        """
        Refresh the file list for a specific arg_name's GUI frame.
        """
        dir_path = os.path.join(name_of_folder, 'saved_deigo', arg_data.arg_name)
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            files.sort()
            arg_data.files_listbox.delete(0, tk.END)
            for f in files:
                arg_data.files_listbox.insert(tk.END, f)
            arg_data.files_listbox.config(height=min(len(files), 10))
        else:
            arg_data.files_listbox.delete(0, tk.END)
            arg_data.files_listbox.config(height=1)
        check_files_for_arg_name_plot_dict(arg_data)

    def create_files(self, arg_data):
        """
        Create plot dict ping files for a specific arg_name.
        """
        create_files_for_arg_name_plot_dict(arg_data)

    def save_agent(self, arg_data):
        """
        Create agent save ping files for a specific arg_name.
        """
        create_files_for_arg_name_save_agent(arg_data)

    def delete_files(self, arg_data):
        """
        Delete files for a specific arg_name.
        """
        dir_path = os.path.join(name_of_folder, 'saved_deigo', arg_data.arg_name)
        if os.path.exists(dir_path):
            for item in os.listdir(dir_path):
                if item != 'agents':
                    item_path = os.path.join(dir_path, item)
                    try:
                        if os.path.isfile(item_path) or os.path.islink(item_path):
                            os.unlink(item_path)
                        elif os.path.isdir(item_path):
                            shutil.rmtree(item_path)
                    except Exception as e:
                        print(f'Failed to delete {item_path}. Reason: {e}')
            self.update_arg_name_frame(arg_data)


if __name__ == '__main__':
    root = tk.Tk()
    app = GUIApp(root)
    root.mainloop()
# %%
