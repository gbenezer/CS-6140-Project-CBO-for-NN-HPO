import logging
from ax.service.ax_client import AxClient, ObjectiveProperties

# hopefully gets rid of printed output
logging.getLogger("ax.service").setLevel(logging.WARNING)
logging.getLogger("ax.modelbridge").setLevel(logging.WARNING)
logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.WARNING)
logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
logging.getLogger("ax.service.ax_client").setLevel(logging.WARNING)

# Status quo parameter settings that get really good results
# (benchmark configurations)
MNIST_status_quo_parameters = {
    "input_dropout_probability": 0.1,
    "hidden_dropout_probability": 0.5,
    "output_dropout_probability": 0.1,
    "hidden_layer_nodes_1": 500,
    "hidden_layer_nodes_2": 100,
    "hidden_layer_nodes_3": 50,
    "activation": "relu",
    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "w_decay": 0,
}
Superconductivity_status_quo_parameters = {
    "input_dropout_probability": 0.1,
    "hidden_dropout_probability": 0.2,
    "output_dropout_probability": 0.1,
    "hidden_layer_nodes_1": 27,
    "hidden_layer_nodes_2": 9,
    "hidden_layer_nodes_3": 3,
    "activation": "relu",
    "learning_rate": 1e-3,
    "beta1": 0.9,
    "beta2": 0.999,
    "w_decay": 0,
}

# Experimental parameter bounds
MNIST_parameters = [
    {
        "name": "input_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "output_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_1",
        "type": "range",
        "bounds": [20, 1000],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_2",
        "type": "range",
        "bounds": [20, 1000],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_3",
        "type": "range",
        "bounds": [20, 1000],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-8, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "beta1",
        "type": "range",
        "bounds": [1e-8, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "beta2",
        "type": "range",
        "bounds": [1e-8, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "w_decay",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "activation",
        "type": "choice",
        "values": ["swish", "sigmoid", "relu", "leaky_relu"],
        "value_type": "str",
        "is_ordered": False,
        "sort_values": False,
    },
]

Superconductivity_parameters = [
    {
        "name": "input_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "output_dropout_probability",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_1",
        "type": "range",
        "bounds": [10, 200],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_2",
        "type": "range",
        "bounds": [10, 200],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "hidden_layer_nodes_3",
        "type": "range",
        "bounds": [10, 200],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "learning_rate",
        "type": "range",
        "bounds": [1e-8, 1.0],
        "value_type": "float",
        "log_scale": True,
    },
    {
        "name": "beta1",
        "type": "range",
        "bounds": [1e-8, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "beta2",
        "type": "range",
        "bounds": [1e-8, (1 - (1e-8))],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "w_decay",
        "type": "range",
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "activation",
        "type": "choice",
        "values": ["swish", "sigmoid", "relu", "leaky_relu"],
        "value_type": "str",
        "is_ordered": False,
        "sort_values": False,
    },
]

# definition and bounds for budget variables just in case
budget_variables = [
    {
        "name": "max_epochs",
        "type": "range",
        "bounds": [1, 100],
        "value_type": "int",
        "log_scale": False,
    },
    {
        "name": "fraction_train_data",
        "type": "range",
        "bounds": [0.1, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
]

# Single objective and multiobjective definitions
MNIST_single_objective = {"test_accuracy": ObjectiveProperties(minimize=False)}

Superconductivity_single_objective = {
    "test_nrmse_mean": ObjectiveProperties(minimize=True)
}

# TODO: winzorize and/or get thresholds

# For MNIST, valuable configurations have test accuracy at least 90%
# and training time less than 10 minutes
# TODO: set thresholds for other objectives
MNIST_multiobjective = {
    "test_accuracy": ObjectiveProperties(minimize=False, threshold=0.9),
    "number_parameters": ObjectiveProperties(minimize=True),
    "training_time": ObjectiveProperties(minimize=True, threshold=10.0),
    "checkpoint_size": ObjectiveProperties(minimize=True),
}


# For Superconductivity, valuable configurations have test normalized root mean
# squared error (relative to mean RMSE, CV(RMSE)) less than 0.5
# and training time less than 10 minutes
# TODO: set thresholds for other objectives
Superconductivity_multiobjective = {
    "test_nrmse_mean": ObjectiveProperties(minimize=True, threshold=0.5),
    "number_parameters": ObjectiveProperties(minimize=True),
    "training_time": ObjectiveProperties(minimize=True, threshold=10.0),
    "checkpoint_size": ObjectiveProperties(minimize=True),
}

# Parameter constraints (relative to each other; bounds are defined in parameters)
p_constraints = [
    "hidden_layer_nodes_2 <= hidden_layer_nodes_1",
    "hidden_layer_nodes_3 <= hidden_layer_nodes_2",
]

# Outcome constraints
o_constraints = None

# Metrics that are not necessarily objectives but are output by the evaluate hyperparameters function
general_tracking_metrics = [
    "mean_validation_loss",
    "cumulative_validation_loss",
    "mean_test_loss",
    "cumulative_test_loss",
    "testing_time",
    "validation_time"
]
single_objective_added_metrics = [
    "number_parameters",
    "training_time",
    "checkpoint_size",
]
classification_metrics = ["validation_accuracy"]
regression_metrics = [
    "validation_mse",
    "validation_nrmse_mean",
    "validation_nrmse_range",
    "validation_nrmse_std",
    "test_mse",
    "test_nrmse_range",
    "test_nrmse_std",
]

# Combinations for actual use in experiments
classification_tracking_metrics_multi = (
    general_tracking_metrics + classification_metrics
)
regression_tracking_metrics_multi = general_tracking_metrics + regression_metrics


classification_tracking_metrics_single = (
    general_tracking_metrics + classification_metrics + single_objective_added_metrics
)
regression_tracking_metrics_single = (
    general_tracking_metrics + regression_metrics + single_objective_added_metrics
)

# ax_client.create_experiment(
#     name="regression_testing",
#     parameters=Superconductivity_parameters,
#     objectives=Superconductivity_single_objective,
#     parameter_constraints=p_constraints,
#     outcome_constraints=o_constraints,
# )

# # creating a random sampler and testing it
# space_filling_random_sampler = get_sobol(
#     search_space=ax_client.experiment.search_space, seed=0
# )

# random_sample = space_filling_random_sampler.gen(n=20)
# random_sample_parameter_list = [arm.parameters for arm in random_sample.arms]

# # # these arm parameter dictionaries can be unpacked with double asterisks to enable function calls with kwargs
# # for arm in random_sample.arms:
# #     print(arm.parameters)

# creating SearchSpace objects with Service API in case of use for
# random generation of trials with Sobol sampler
ax_client_Super = AxClient()
ax_client_MNIST = AxClient()

ax_client_Super.create_experiment(
    name="Superconductivity_Search_Space_Construction",
    parameters=Superconductivity_parameters,
    objectives=Superconductivity_single_objective,
    parameter_constraints=p_constraints,
    outcome_constraints=o_constraints,
)

ax_client_MNIST.create_experiment(
    name="MNIST_Search_Space_Construction",
    parameters=MNIST_parameters,
    objectives=MNIST_single_objective,
    parameter_constraints=p_constraints,
    outcome_constraints=o_constraints,
)

MNIST_SearchSpace = ax_client_MNIST.experiment.search_space
Superconductivity_SearchSpace = ax_client_Super.experiment.search_space
