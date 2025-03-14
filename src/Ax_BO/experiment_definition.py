import logging
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.factory import get_sobol

# hopefully gets rid of printed output
logging.getLogger("ax.service").setLevel(logging.WARNING)
logging.getLogger("ax.modelbridge").setLevel(logging.WARNING)
logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.WARNING)
logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
logging.getLogger("ax.service.ax_client").setLevel(logging.WARNING)

ax_client = AxClient()

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
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "beta2",
        "type": "range",
        "bounds": [0.0, 1.0],
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
        "bounds": [0.0, 1.0],
        "value_type": "float",
        "log_scale": False,
    },
    {
        "name": "beta2",
        "type": "range",
        "bounds": [0.0, 1.0],
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

MNIST_parameters_w_budget = MNIST_parameters + budget_variables

MNIST_single_objective = {"test_accuracy": ObjectiveProperties(minimize=False)}
Superconductivity_single_objective = {
    "test_nrmse_mean": ObjectiveProperties(minimize=True)
}

MNIST_multiobjective = {
    "test_accuracy": ObjectiveProperties(minimize=False, threshold=0.9),
    "number_parameters": ObjectiveProperties(minimize=True),
    "training_time": ObjectiveProperties(minimize=True),
    "checkpoint_size": ObjectiveProperties(minimize=True),
}

Superconductivity_multiobjective = {
    "test_nrmse_mean": ObjectiveProperties(minimize=True),
    "number_parameters": ObjectiveProperties(minimize=True),
    "training_time": ObjectiveProperties(minimize=True),
    "checkpoint_size": ObjectiveProperties(minimize=True),
}

p_constraints = [
    "hidden_layer_nodes_2 <= hidden_layer_nodes_1",
    "hidden_layer_nodes_3 <= hidden_layer_nodes_2",
]

o_constraints = None

ax_client.create_experiment(
    name="regression_testing",
    parameters=Superconductivity_parameters,
    objectives=Superconductivity_single_objective,
    parameter_constraints=p_constraints,
    outcome_constraints=o_constraints,
)

# creating a random sampler and testing it
space_filling_random_sampler = get_sobol(
    search_space=ax_client.experiment.search_space, seed=0
)

random_sample = space_filling_random_sampler.gen(n=20)
random_sample_parameter_list = [arm.parameters for arm in random_sample.arms]

# # these arm parameter dictionaries can be unpacked with double asterisks to enable function calls with kwargs
# for arm in random_sample.arms:
#     print(arm.parameters)
