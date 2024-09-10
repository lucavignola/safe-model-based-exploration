import wandb
import pandas as pd

# Define your project name. Replace 'your-username/your-project' with your specific details
ENTITY = 'trevenl'
PROJECT = 'ExplorationPendulum13hSep092024'

# Initiate the API
api = wandb.Api()

# Fetch all runs for the project
runs = api.runs(f"{ENTITY}/{PROJECT}")

# List to hold the data
data = []

# Loop through each run and collect the config and metrics
for run in runs:
    # Fetch run id, config, and summary metrics
    run_id = run.id
    config = run.config
    summary = run.summary

    # Add the run_id to the config and summary dictionaries
    config['run_id'] = run_id
    summary_dict = {f"summary_{key}": value for key, value in summary.items()}
    summary_dict['run_id'] = run_id

    # Combine config and summary into one record
    record = {**config, **summary_dict}

    # Add to the data list
    data.append(record)

# Convert the data list to a pandas DataFrame
df = pd.DataFrame(data)

# Save the DataFrame to a CSV file
df.to_csv(f'{PROJECT}.csv', index=False)

print("Data saved to 'wandb_runs.csv'")
