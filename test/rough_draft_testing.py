# standard imports
import torch.nn as nn
import logging
from multiprocessing import freeze_support
import os
from pathlib import Path
from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.exceptions.core import OptimizationShouldStop
import pandas as pd
from ax.service.utils.report_utils import _pairwise_pareto_plotly_scatter
import plotly.io as pio

# acquisition functions
from botorch.acquisition.analytic import (
    ExpectedImprovement,
    LogExpectedImprovement,
    LogNoisyExpectedImprovement,
    NoisyExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.logei import (
    qLogExpectedImprovement,
    qLogNoisyExpectedImprovement,
)
from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient

from botorch.acquisition.multi_objective.logei import (
    qLogNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.hypervolume_knowledge_gradient import (
    qHypervolumeKnowledgeGradient,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# # importing the correct package defined functions
from src.network.create_network_lightning import create_ff_model
from src.network.load_data import get_MNIST_data, get_Superconductivity_data
from src.network.evaluate_network import evaluate_hyperparameters
from src.Ax_BO.conduct_experiment import conduct_experiment
import src.Ax_BO.experiment_definition as exp_def

# evaluate_hyperparameters testing and benchmark creation

if __name__ == "__main__":
    freeze_support()

    # remove sanity check and other extraneous stout printing
    # logging.getLogger("lightning").setLevel(logging.WARNING)
    # logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    # logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.WARNING)
    # logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
    # logging.getLogger("ax.service.ax_client").setLevel(logging.WARNING)

    # Data Loading

    (
        trainset_MNIST,
        validset_MNIST,
        testset_MNIST,
        trainloader_MNIST,
        validloader_MNIST,
        testloader_MNIST,
    ) = get_MNIST_data(
        valid_fraction=0.2,
        random_seed=42,
        n_workers=15,
        batch_n=64,
        download_data=False,
    )

    (
        fulldataset_Super,
        trainset_Super,
        validset_Super,
        testset_Super,
        trainloader_Super,
        validloader_Super,
        testloader_Super,
    ) = get_Superconductivity_data(
        valid_fraction=0.2,
        test_fraction=0.2,
        random_seed=0,
        n_workers=15,
        batch_n=20,
        local=False,
    )

    # Experiment code testing

    # # random classification, single objective
    # MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
    #     conduct_experiment(
    #         task="classification",
    #         parameter_space=exp_def.MNIST_parameters,
    #         objective=exp_def.MNIST_single_objective,
    #         param_constraints=exp_def.p_constraints,
    #         out_constraints=exp_def.o_constraints,
    #         tracking_metrics=exp_def.classification_tracking_metrics_single,
    #         acquisition_func_class=qLogNoisyExpectedImprovement,
    #         train_loader=trainloader_MNIST,
    #         valid_loader=validloader_MNIST,
    #         test_loader=testloader_MNIST,
    #         input_shape=(1, 28, 28),
    #         number_input_features=784,
    #         number_output_features=10,
    #         loss=nn.CrossEntropyLoss(),
    #         max_trials=100,
    #         num_reps_per_trial=1,
    #         max_epochs=5,
    #         experiment_name="Random_MNIST_2",
    #         global_early_stop=True,
    #         fully_random=True,
    #         interactive_plots=False,
    #         seed=None,
    #     )
    # )

    # # random regression, single objective
    # Super_experiment_df, Super_client_object, Super_experiment_object = (
    #     conduct_experiment(
    #         task="regression",
    #         parameter_space=exp_def.Superconductivity_parameters,
    #         objective=exp_def.Superconductivity_single_objective,
    #         param_constraints=exp_def.p_constraints,
    #         out_constraints=exp_def.o_constraints,
    #         tracking_metrics=exp_def.regression_tracking_metrics_single,
    #         acquisition_func_class=qLogNoisyExpectedImprovement,
    #         train_loader=trainloader_Super,
    #         valid_loader=validloader_Super,
    #         test_loader=testloader_Super,
    #         input_shape=(1, 1, 81),
    #         number_input_features=81,
    #         number_output_features=1,
    #         loss=nn.HuberLoss(),
    #         max_trials=100,
    #         num_reps_per_trial=1,
    #         max_epochs=5,
    #         experiment_name="Random_Super_1",
    #         global_early_stop=True,
    #         fully_random=True,
    #         interactive_plots=False,
    #         seed=None,
    #     )
    # )

    # non-random classification, single objective
    MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
        conduct_experiment(
            task="classification",
            parameter_space=exp_def.MNIST_parameters,
            objective=exp_def.MNIST_single_objective,
            param_constraints=exp_def.p_constraints,
            out_constraints=exp_def.o_constraints,
            tracking_metrics=exp_def.classification_tracking_metrics_single,
            acquisition_func_class=qLogNoisyExpectedImprovement,
            train_loader=trainloader_MNIST,
            valid_loader=validloader_MNIST,
            test_loader=testloader_MNIST,
            input_shape=(1, 28, 28),
            number_input_features=784,
            number_output_features=10,
            loss=nn.CrossEntropyLoss(),
            max_trials=50,
            num_reps_per_trial=1,
            max_epochs=5,
            experiment_name="Random_MNIST_50_3",
            global_early_stop=False,
            fully_random=True,
            interactive_plots=False,
            seed=None,
        )
    )
    
    # non-random regression, single objective
    Super_experiment_df, Super_client_object, Super_experiment_object = (
        conduct_experiment(
            task="regression",
            parameter_space=exp_def.Superconductivity_parameters,
            objective=exp_def.Superconductivity_single_objective,
            param_constraints=exp_def.p_constraints,
            out_constraints=exp_def.o_constraints,
            tracking_metrics=exp_def.regression_tracking_metrics_single,
            acquisition_func_class=qLogNoisyExpectedImprovement,
            train_loader=trainloader_Super,
            valid_loader=validloader_Super,
            test_loader=testloader_Super,
            input_shape=(1, 1, 81),
            number_input_features=81,
            number_output_features=1,
            loss=nn.HuberLoss(),
            max_trials=50,
            num_reps_per_trial=1,
            max_epochs=5,
            experiment_name="Random_Super_50_3",
            global_early_stop=False,
            fully_random=True,
            interactive_plots=False,
            seed=None,
        )
    )
    
    # non-random classification, single objective
    MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
        conduct_experiment(
            task="classification",
            parameter_space=exp_def.MNIST_parameters,
            objective=exp_def.MNIST_single_objective,
            param_constraints=exp_def.p_constraints,
            out_constraints=exp_def.o_constraints,
            tracking_metrics=exp_def.classification_tracking_metrics_single,
            acquisition_func_class=qLogNoisyExpectedImprovement,
            train_loader=trainloader_MNIST,
            valid_loader=validloader_MNIST,
            test_loader=testloader_MNIST,
            input_shape=(1, 28, 28),
            number_input_features=784,
            number_output_features=10,
            loss=nn.CrossEntropyLoss(),
            max_trials=50,
            num_reps_per_trial=1,
            max_epochs=5,
            experiment_name="qLogNEI_MNIST_2",
            global_early_stop=False,
            fully_random=False,
            interactive_plots=True,
            seed=None,
        )
    )
    
    # non-random regression, single objective
    Super_experiment_df, Super_client_object, Super_experiment_object = (
        conduct_experiment(
            task="regression",
            parameter_space=exp_def.Superconductivity_parameters,
            objective=exp_def.Superconductivity_single_objective,
            param_constraints=exp_def.p_constraints,
            out_constraints=exp_def.o_constraints,
            tracking_metrics=exp_def.regression_tracking_metrics_single,
            acquisition_func_class=qLogNoisyExpectedImprovement,
            train_loader=trainloader_Super,
            valid_loader=validloader_Super,
            test_loader=testloader_Super,
            input_shape=(1, 1, 81),
            number_input_features=81,
            number_output_features=1,
            loss=nn.HuberLoss(),
            max_trials=50,
            num_reps_per_trial=1,
            max_epochs=5,
            experiment_name="qLogNEI_Super_2",
            global_early_stop=False,
            fully_random=False,
            interactive_plots=True,
            seed=None,
        )
    )

    # # non-random classification, multi-objective
    # MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
    #     conduct_experiment(
    #         task="classification",
    #         parameter_space=exp_def.MNIST_parameters,
    #         objective=exp_def.MNIST_multiobjective,
    #         param_constraints=exp_def.p_constraints,
    #         out_constraints=exp_def.o_constraints,
    #         tracking_metrics=exp_def.classification_tracking_metrics_multi,
    #         acquisition_func_class=qLogNoisyExpectedHypervolumeImprovement,
    #         train_loader=trainloader_MNIST,
    #         valid_loader=validloader_MNIST,
    #         test_loader=testloader_MNIST,
    #         input_shape=(1, 28, 28),
    #         number_input_features=784,
    #         number_output_features=10,
    #         loss=nn.CrossEntropyLoss(),
    #         max_trials=100,
    #         num_reps_per_trial=1,
    #         max_epochs=5,
    #         experiment_name="qLogNoisyExpectedHypervolumeImprovement_MNIST_1",
    #         global_early_stop=False,
    #         fully_random=False,
    #         interactive_plots=False,
    #         seed=None,
    #     )
    # )
    
    # # non-random classification, single objective
    # MNIST_experiment_df, MNIST_client_object, MNIST_experiment_object = (
    #     conduct_experiment(
    #         task="classification",
    #         parameter_space=exp_def.MNIST_parameters,
    #         objective=exp_def.MNIST_single_objective,
    #         param_constraints=exp_def.p_constraints,
    #         out_constraints=exp_def.o_constraints,
    #         tracking_metrics=exp_def.classification_tracking_metrics_single,
    #         acquisition_func_class=qLogNoisyExpectedImprovement,
    #         train_loader=trainloader_MNIST,
    #         valid_loader=validloader_MNIST,
    #         test_loader=testloader_MNIST,
    #         input_shape=(1, 28, 28),
    #         number_input_features=784,
    #         number_output_features=10,
    #         loss=nn.CrossEntropyLoss(),
    #         max_trials=100,
    #         num_reps_per_trial=1,
    #         max_epochs=5,
    #         experiment_name="qLogNoisyExpectedImprovement_MNIST_1",
    #         global_early_stop=True,
    #         fully_random=False,
    #         interactive_plots=True,
    #         seed=None,
    #     )
    # )

    # # non-random regression, multi-objective
    # Super_experiment_df, Super_client_object, Super_experiment_object = (
    #     conduct_experiment(
    #         task="regression",
    #         parameter_space=exp_def.Superconductivity_parameters,
    #         objective=exp_def.Superconductivity_multiobjective,
    #         param_constraints=exp_def.p_constraints,
    #         out_constraints=exp_def.o_constraints,
    #         tracking_metrics=exp_def.regression_tracking_metrics_multi,
    #         acquisition_func_class=qLogNoisyExpectedHypervolumeImprovement,
    #         train_loader=trainloader_Super,
    #         valid_loader=validloader_Super,
    #         test_loader=testloader_Super,
    #         input_shape=(1, 1, 81),
    #         number_input_features=81,
    #         number_output_features=1,
    #         loss=nn.HuberLoss(),
    #         max_trials=100,
    #         num_reps_per_trial=1,
    #         max_epochs=5,
    #         experiment_name="qLogNoisyExpectedHypervolumeImprovement_Super_1",
    #         global_early_stop=False,
    #         fully_random=False,
    #         interactive_plots=False,
    #         seed=None,
    #     )
    # )
