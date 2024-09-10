import os
import wandb

# Initialize wandb API
api = wandb.Api()

# Replace 'your_project_name' with your actual project name
PROJECT_NAME = 'ExplorationPendulum13hSep092024'
# Replace 'your_entity' if your project is under a team. Skip if it's under your user.
ENTITY = 'trevenl'

# Fetch the list of runs for the project
runs = api.runs(f"{ENTITY}/{PROJECT_NAME}")

# Local structure creation
os.makedirs(PROJECT_NAME, exist_ok=True)

for run in runs:
    run_id = run.id
    run_folder = os.path.join(PROJECT_NAME, run_id)
    os.makedirs(run_folder, exist_ok=True)

    # Fetch the files in the run
    files = run.files()
    print(f'Starting with run {run_id}')

    for file in files:
        if file.name.startswith('saved_data/'):
            file.download(root=run_folder)

print('Downloaded all files successfully!')