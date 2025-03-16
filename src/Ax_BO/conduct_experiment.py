# function for conducting a given experiment

# standard imports
from pathlib import Path
import os
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import ModelRegistryBase, Models
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.ax_client import AxClient, ObjectiveProperties


from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.slice import plot_slice
import plotly.io as pio

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

from src.network.evaluate_network import evaluate_hyperparameters

# TODO: figure out how to add tracking metrics for the other metrics I am tracking
# with AxClient.add_tracking_metrics
def conduct_experiment(
    task,
    parameter_space,
    search_space,
    objective,
    param_constraints,
    out_constraints,
    tracking_metrics,
    train_loader,
    valid_loader,
    test_loader,
    input_shape,
    number_input_features,
    number_output_features,
    loss,
    max_trials,
    num_reps_per_trial,
    max_epochs,
    experiment_name,
    fully_random,
    seed,
):

    # making the plot directory
    plot_directory = Path(os.getcwd()) / "plots/html_plots" / experiment_name
    if not os.path.exists(plot_directory):
        plot_directory.mkdir(parents=True)

    # if the trial is supposed to be fully random, force the AxClient to use fully random search
    if fully_random:
        gen_strat = choose_generation_strategy(
            search_space=search_space,
            random_seed=seed,
            force_random_search=fully_random,
        )
    else:
        gen_strat = None

    ax_client = AxClient(generation_strategy=gen_strat, random_seed=seed)

    # create the experiment
    ax_client.create_experiment(
        name=experiment_name,
        parameters=parameter_space,
        objectives=objective,
        parameter_constraints=param_constraints,
        outcome_constraints=out_constraints,
    )
    
    ax_client.add_tracking_metrics(tracking_metrics)

    # run the optimization loop
    # TODO: implement logic to switch out acquisition functions
    for _ in range(max_trials):
        parameterization, trial_index = ax_client.get_next_trial()
        trial_log_dir_name = experiment_name + "/" + f"trial_{trial_index}"
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data=evaluate_hyperparameters(
                task=task,
                train_loader=train_loader,
                valid_loader=valid_loader,
                test_loader=test_loader,
                input_shape=input_shape,
                number_input_features=number_input_features,
                number_output_features=number_output_features,
                loss=loss,
                log_dir_name=trial_log_dir_name,
                num_rep=num_reps_per_trial,
                max_epochs=max_epochs,
                parameterization=parameterization,
            ),
        )

    # getting results of experiment
    experiment_df = ax_client.experiment.to_df()
    best_parameters, values = ax_client.get_best_parameters()
    
    # saving the results of the experiment to a CSV file
    experiment_df.to_csv(Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs" / (experiment_name + ".csv"))


    # saving interactive plots
    model = ax_client.generation_strategy.model

    # TODO: make this configurable
    plotly_metric_contour = interact_contour(
        model=model, metric_name="test_accuracy"
    ).data
    cv_results = cross_validate(model)
    plotly_tradeoff = plot_objective_vs_constraints(
        model, "test_accuracy", rel=False
    ).data
    plotly_cv = interact_cross_validation(cv_results).data
    plotly_tile = interact_fitted(model, rel=False).data

    pio.write_html(
        fig=plotly_metric_contour,
        file=plot_directory / "example_test_accuracy_interactive_contour.html",
    )
    pio.write_html(
        fig=plotly_tradeoff,
        file=plot_directory / "example_tradeoff_interactive_plot.html",
    )
    pio.write_html(fig=plotly_cv, file=plot_directory / "example_cv_plot.html")
    pio.write_html(fig=plotly_tile, file=plot_directory / "example_tile_plot.html")

    return experiment_df, best_parameters, values


import logging
from multiprocessing import freeze_support
import torch.nn as nn
from src.network.load_data import get_MNIST_data, get_Superconductivity_data
from src.Ax_BO.experiment_definition import (
    MNIST_parameters,
    MNIST_SearchSpace,
    MNIST_single_objective,
    p_constraints,
    o_constraints,
    classification_tracking_metrics_single
)

from ax.global_stopping.strategies.improvement import ImprovementGlobalStoppingStrategy
from ax.exceptions.core import OptimizationShouldStop

if __name__ == "__main__":
    freeze_support()

    # remove sanity check and other extraneous stout printing
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.WARNING)
    logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
    logging.getLogger("ax.service.ax_client").setLevel(logging.WARNING)

    train_set, valid_set, testset, trainloader, validloader, testloader = (
        get_MNIST_data(
            valid_fraction=0.2,
            random_seed=42,
            n_workers=15,
            batch_n=64,
            download_data=False,
        )
    )

    experiment_df, best_parameters, values = conduct_experiment(
        task="classification",
        parameter_space=MNIST_parameters,
        search_space=MNIST_SearchSpace,
        objective=MNIST_single_objective,
        param_constraints=p_constraints,
        out_constraints=o_constraints,
        tracking_metrics=classification_tracking_metrics_single,
        train_loader=trainloader,
        valid_loader=validloader,
        test_loader=testloader,
        input_shape=(1, 28, 28),
        number_input_features=784,
        number_output_features=10,
        loss=nn.CrossEntropyLoss(),
        max_trials=4,
        num_reps_per_trial=1,
        max_epochs=5,
        experiment_name="MNIST_test",
        fully_random=False,
        seed=0,
    )
    print(experiment_df)
    print(best_parameters)
    print(values)
    experiment_df.to_csv("test_experiment")
