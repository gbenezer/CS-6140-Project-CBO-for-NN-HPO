# function for conducting a given experiment

# standard imports
from pathlib import Path
import os
from typing import Literal
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.service.ax_client import AxClient
from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy, ThresholdEarlyStoppingStrategy
from ax.global_stopping.strategies import ImprovementGlobalStoppingStrategy
from ax.service.utils.instantiation import InstantiationBase
from ax.exceptions.core import OptimizationShouldStop
from ax.metrics.tensorboard import TensorboardMetric
from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer


from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints
from ax.modelbridge.cross_validation import cross_validate
import plotly.io as pio

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

from src.network.evaluate_network import evaluate_hyperparameters
from src.Ax_BO.threshold_global_stopping import ThresholdGlobalStoppingStrategy

def conduct_experiment_developer(
    task: Literal["regression", "classification"],
    parameter_space,
    objective,
    param_constraints,
    out_constraints,
    tracking_metrics,
    acquisition_func_class,
    train_loss_threshold,
    test_loss_threshold,
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
    interactive_plots,
    global_early_stop,
    trial_early_stop,
    seed,
):

    # making the plot directory
    plot_directory = Path(os.getcwd()) / "plots/html_plots" / experiment_name
    if not os.path.exists(plot_directory):
        plot_directory.mkdir(parents=True)

    # the minimum number of initialization trials should be twice the parameter space dimension
    # (guideline heuristic in BoTorch is max(5, 2 * len(parameter_space))), and all the parameter spaces
    # I am looking at are higher dimensional by a factor of 2
    min_number_initialization_trials = 2 * len(parameter_space)

    # if the trial is supposed to be fully random, force the AxClient to only use fully quasi-random search
    if fully_random:
        gen_strat = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,  # quasi-random space filling generation
                    num_trials=-1,  # unlimited number of trials can be generated
                )
            ],
            name="quasi_random_Sobol",
        )
    else:
        gen_strat = GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,  # quasi-random space filling generation
                    num_trials=min_number_initialization_trials,  # minimum and maximum are the same for this strategy
                    min_trials_observed=min_number_initialization_trials,
                ),
                GenerationStep(
                    model=Models.BOTORCH_MODULAR,
                    num_trials=-1,  # unlimited
                    model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
                        "botorch_acqf_class": acquisition_func_class,
                    },
                ),
            ],
            name="custom_acquisition_function_and_model",
        )

    # TODO: allow configuration of threshold parameters
    # NOTE: Global early stopping will break for multi-objective problems
    if global_early_stop and task == "classification":
        # 95% test accuracy threshold
        gss = ThresholdGlobalStoppingStrategy(min_trials=5, threshold=0.95)
    elif global_early_stop and task == "regression":
        # 0.2 range-normalized root mean square deviation
        gss = ThresholdGlobalStoppingStrategy(min_trials=5, threshold=0.2)
    else:
        gss = None
        
    if trial_early_stop:
        # good threshold for MNIST looks to be 5,
        # yet to be determined for Superconductivity
        ess = ThresholdEarlyStoppingStrategy(metric_names=["training_loss"],
                                             metric_threshold=train_loss_threshold,
                                             min_progression=0.1,
                                             min_curves=1,
                                             normalize_progressions=True)
    else:
        ess = None

    # defining the SearchSpace
    search_space = InstantiationBase.make_search_space(parameters=parameter_space,
                                                       parameter_constraints=param_constraints)
    
    # defining the OptimizationConfig
    optimization_config = InstantiationBase.make_optimization_config_from_properties(objectives=objective,
                                                                                     outcome_constraints=out_constraints)

    # TODO: finish code
    # ax_client = AxClient(
    #     generation_strategy=gen_strat,
    #     random_seed=seed,
    #     global_stopping_strategy=gss,
    #     early_stopping_strategy=ess
    # )
    
    # # ax_client.should_stop_trials_early

    # # create the experiment
    # if trial_early_stop:
    #     ax_client.create_experiment(
    #         name=experiment_name,
    #         parameters=parameter_space,
    #         objectives=objective,
    #         parameter_constraints=param_constraints,
    #         outcome_constraints=out_constraints,
    #         support_intermediate_data=True
    #     )
    # else:
    #     ax_client.create_experiment(
    #         name=experiment_name,
    #         parameters=parameter_space,
    #         objectives=objective,
    #         parameter_constraints=param_constraints,
    #         outcome_constraints=out_constraints
    #     )

    # ax_client.add_tracking_metrics(tracking_metrics)
    
    # # method to add TensorBoard metric for early stopping
    # if trial_early_stop:
    #     ax_client.experiment.add_tracking_metric(TensorboardMetric(name="training_loss",
    #                                                             tag="training_loss",
    #                                                             lower_is_better=True))

    # # run the optimization loop
    # for _ in range(max_trials):
        
    #     try:
    #         parameterization, trial_index = ax_client.get_next_trial()
    #     except OptimizationShouldStop as exc:
    #         print(exc.message)
    #         break
        
    #     trial_log_dir_name = experiment_name + "/" + f"trial_{trial_index}"
    #     trial_data = evaluate_hyperparameters(
    #             task=task,
    #             train_loader=train_loader,
    #             valid_loader=valid_loader,
    #             test_loader=test_loader,
    #             input_shape=input_shape,
    #             number_input_features=number_input_features,
    #             number_output_features=number_output_features,
    #             loss=loss,
    #             log_dir_name=trial_log_dir_name,
    #             num_rep=num_reps_per_trial,
    #             max_epochs=max_epochs,
    #             parameterization=parameterization,
    #     )
        
    #     # test whether or not there are absurd test loss values
    #     if num_reps_per_trial > 1:
    #         if trial_data["mean_test_loss"][0] > test_loss_threshold:
    #             ax_client.abandon_trial(trial_index=trial_index, reason="mean test loss too high")
    #             continue
    #     elif trial_data["mean_test_loss"] > test_loss_threshold:
    #         ax_client.abandon_trial(trial_index=trial_index, reason="mean test loss too high")
    #         continue
        
    #     # TODO: test SEM toggle output because of this
    #     # NOTE: When ``raw_data`` does not specify SEM for a given metric, Ax
    #     # will default to the assumption that the data is noisy (specifically,
    #     # corrupted by additive zero-mean Gaussian noise) and that the
    #     # level of noise should be inferred by the optimization model.
        
    #     print(trial_data)
        
    #     try:
    #         ax_client.complete_trial(
    #             trial_index=trial_index,
    #             raw_data=trial_data,
    #         )
    #     except ValueError:
    #         ax_client.abandon_trial(trial_index=trial_index, reason="invalid metric values")
    #         continue

    # # getting results of experiment
    # experiment_df = ax_client.experiment.to_df()
    # best_parameters, values = ax_client.get_best_parameters()

    # # saving the results of the experiment to a CSV file
    # experiment_df_dir = Path(os.getcwd()) / "logs" / "csv_logs" / "experiment_logs"
    # if not os.path.exists(experiment_df_dir):
    #     experiment_df_dir.mkdir(parents=True)
    # experiment_csv_name = experiment_name + ".csv"
    # experiment_df.to_csv(path_or_buf=(experiment_df_dir / experiment_csv_name))

    # if interactive_plots:
    #     # saving interactive plots (only works if there are non-RandomBridgeModel trials completed)
    #     model = ax_client.generation_strategy.model

    #     # TODO: make this configurable
    #     plotly_metric_contour = interact_contour(
    #         model=model, metric_name="test_accuracy"
    #     ).data
    #     cv_results = cross_validate(model)
    #     plotly_tradeoff = plot_objective_vs_constraints(
    #         model, "test_accuracy", rel=False
    #     ).data
    #     plotly_cv = interact_cross_validation(cv_results).data
    #     plotly_tile = interact_fitted(model, rel=False).data

    #     pio.write_html(
    #         fig=plotly_metric_contour,
    #         file=plot_directory / "example_test_accuracy_interactive_contour.html",
    #     )
    #     pio.write_html(
    #         fig=plotly_tradeoff,
    #         file=plot_directory / "example_tradeoff_interactive_plot.html",
    #     )
    #     pio.write_html(fig=plotly_cv, file=plot_directory / "example_cv_plot.html")
    #     pio.write_html(fig=plotly_tile, file=plot_directory / "example_tile_plot.html")

    # return experiment_df, best_parameters, values
    pass