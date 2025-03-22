import os
import pandas as pd
from pathlib import Path

experiment_directory = Path(os.getcwd())/ "logs" / "csv_logs" / "experiment_logs"

experiment_data = pd.read_csv(filepath_or_buffer=(experiment_directory / "qLogNoisyExpectedImprovement_Test_1.csv"))
print(experiment_data["checkpoint_size"].describe())