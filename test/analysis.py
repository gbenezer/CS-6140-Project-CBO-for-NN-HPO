import os
import pandas as pd
from pathlib import Path
import numpy as np
import plotly.express as px
import plotly.io as pio
import sys
from ax.service.utils.instantiation import InstantiationBase

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")
import src.Ax_BO.experiment_definition as exp_def
from ax import (
    ChoiceParameter,
    ComparisonOp,
    Experiment,
    FixedParameter,
    Metric,
    Objective,
    OptimizationConfig,
    OrderConstraint,
    OutcomeConstraint,
    ParameterType,
    RangeParameter,
    SearchSpace,
    SumConstraint,
)

from ax.modelbridge.registry import Models

experiment_directory = Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs"
plot_directory = Path(os.getcwd()) / "plots" / "html_plots"

MNIST_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "classification_benchmark_results.csv")
)
MNIST_benchmark = MNIST_benchmark.assign(trial_index=-1, dataset="benchmark")
MNIST_random = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "Random_MNIST_1_official.csv")
)
MNIST_random = MNIST_random.assign(dataset="random")
MNIST_LogEI = pd.read_csv(
    filepath_or_buffer=(
        experiment_directory / "qLogNoisyExpectedImprovement_MNIST_Test_1.csv"
    )
)
MNIST_LogEI = MNIST_LogEI.assign(dataset="LogNEI")
MNIST_hypervolume = pd.read_csv(
    filepath_or_buffer=(
        experiment_directory
        / "qLogNoisyExpectedHypervolumeImprovement_MNIST_1_official.csv"
    )
)
MNIST_hypervolume = MNIST_hypervolume.assign(dataset="LogNEHVI")

variables = [
    "trial_index",
    "dataset",
    "training_time",
    "test_accuracy",
    "number_parameters",
]

benchmark_subset = MNIST_benchmark[variables]
random_subset = MNIST_random[variables]
LogEI_subset = MNIST_LogEI[variables]
LogHVEI_subset = MNIST_hypervolume[variables]

joint_df = pd.concat([benchmark_subset, random_subset, LogEI_subset, LogHVEI_subset])

# fig = px.scatter_3d(
#     data_frame=joint_df,
#     x="training_time",
#     y="number_parameters",
#     z="test_accuracy",
#     color="trial_index",
#     symbol="dataset",
#     opacity=0.67
# )
# fig.update_traces(marker_size=5)
# pio.write_html(fig=fig,
#                file=(plot_directory / "MNIST_objective_plot_2.html"))



Super_benchmark = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "regression_benchmark_results.csv")
)
Super_benchmark = Super_benchmark.assign(trial_index=-1, dataset="benchmark")
Super_random = pd.read_csv(
    filepath_or_buffer=(experiment_directory / "Random_Super_1_official.csv")
)
Super_random = Super_random.assign(dataset="random")

Super_LogEI = pd.read_csv(
    filepath_or_buffer=(
        experiment_directory / "qLogNoisyExpectedImprovement_Super_1_official.csv"
    )
)
Super_LogEI = Super_LogEI.assign(dataset="LogNEI")

variables = [
    "trial_index",
    "dataset",
    "training_time",
    "test_nrmse_range",
    "number_parameters",
]

benchmark_subset = Super_benchmark[variables]
random_subset = Super_random[variables]
LogEI_subset = Super_LogEI[variables]


joint_df = pd.concat([benchmark_subset, random_subset, LogEI_subset])

fig = px.scatter_3d(
    data_frame=joint_df,
    x="training_time",
    y="number_parameters",
    z="test_nrmse_range",
    color="trial_index",
    symbol="dataset",
    opacity=0.67
)
fig.update_traces(marker_size=5)
# pio.write_html(fig=fig,
#                file=(plot_directory / "Super_objective_plot_2.html"))

# MNIST_SearchSpace = InstantiationBase().make_search_space(parameters=exp_def.MNIST_parameters,
#                                                           parameter_constraints=exp_def.p_constraints)
# Superconductivity_SearchSpace = InstantiationBase().make_search_space(parameters=exp_def.Superconductivity_parameters,
#                                                           parameter_constraints=exp_def.p_constraints)
# MNIST_Single_Optimization_Config = InstantiationBase().make_optimization_config_from_properties(objectives=exp_def.MNIST_single_objective)
# MNIST_Multiple_Optimization_Config = InstantiationBase().make_optimization_config_from_properties(objectives=exp_def.MNIST_multiobjective)
# Super_Single_Optimization_Config = InstantiationBase().make_optimization_config_from_properties(objectives=exp_def.Superconductivity_single_objective)
# Super_Multiple_Optimization_Config = InstantiationBase().make_optimization_config_from_properties(objectives=exp_def.MNIST_multiobjective)
