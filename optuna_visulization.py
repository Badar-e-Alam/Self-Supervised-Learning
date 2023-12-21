import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import optuna
from optuna.visualization import plot_parallel_coordinate
from optuna import create_study, Trial

# Load the results from the JSON file
with open("optuna_logging.json", "r") as file:
    data = json.load(file)

# Create a study object and add the trials to it
study = create_study(direction="maximize")
study = create_study(direction="minimize")
for trial_data in data:
    frozen_trial = Trial(study=study, state=optuna.structs.TrialState.COMPLETE)
    frozen_trial._trial_id = len(study.trials) + 1
    frozen_trial.params = trial_data[0]["params"]
    frozen_trial.value = trial_data[0]["value"]
    study._append_trial(frozen_trial)  # Add the trial to the study

# Visualize the results using plot_parallel_coordinate()
vis.plot_parallel_coordinate(study)  # Add the trial to the study

# Visualize the results using plot_parallel_coordinate()
plot_parallel_coordinate(study)
