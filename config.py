"""Single source of truth for project paths and cluster settings. Edit config.json, not this."""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
FOLDER_NAME  = PROJECT_ROOT.name
LAUNCH_DIR   = PROJECT_ROOT.parent          # where you run sbatch and where the .sif lives

with open(PROJECT_ROOT / 'config.json') as f:
    CONFIG = json.load(f)

SIF_FILE           = CONFIG['sif_file']
MAX_AGENTS_PER_JOB = CONFIG['max_agents_per_job']
CLUSTERS           = CONFIG['clusters']