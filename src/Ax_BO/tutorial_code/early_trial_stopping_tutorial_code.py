import sys
import plotly.io as pio

import os
import tempfile

from pathlib import Path

import torchx
from multiprocessing import freeze_support

from ax.core import Experiment, Objective, ParameterType, RangeParameter, SearchSpace
from ax.core.optimization_config import OptimizationConfig

from ax.early_stopping.strategies import PercentileEarlyStoppingStrategy
from ax.metrics.tensorboard import TensorboardMetric

from ax.modelbridge.dispatch_utils import choose_generation_strategy

from ax.runners.torchx import TorchXRunner

from ax.service.scheduler import Scheduler, SchedulerOptions
from ax.service.utils.report_utils import exp_to_df

from tensorboard.backend.event_processing import plugin_event_multiplexer as event_multiplexer

from torchx import specs
from torchx.components import utils

from matplotlib import pyplot as plt

SMOKE_TEST = os.environ.get("SMOKE_TEST")

if SMOKE_TEST:
    epochs = 3
else:
    epochs = 10
    
def trainer(
    log_path: str,
    hidden_size_1: int,
    hidden_size_2: int,
    learning_rate: float,
    dropout: float,
    trial_idx: int = -1,
) -> specs.AppDef:

    # define the log path so we can pass it to the TorchX AppDef
    if trial_idx >= 0:
        log_path = Path(log_path).joinpath(str(trial_idx)).absolute().as_posix()

    batch_size = 32
    
    print(log_path)

    return utils.python(
        # command line args to the training script
        "--log_path",
        log_path,
        "--hidden_size_1",
        str(hidden_size_1),
        "--hidden_size_2",
        str(hidden_size_2),
        "--learning_rate",
        str(learning_rate),
        "--epochs",
        str(epochs),
        "--dropout",
        str(dropout),
        "--batch_size",
        str(batch_size),
        # other config options
        name="trainer",
        script="mnist_train_nas.py",
        image=torchx.version.TORCHX_IMAGE,
    )
    
    
if __name__ == "__main__":
    freeze_support()
    # Make a temporary dir to log our results into
    log_dir = tempfile.mkdtemp()
    # print(log_dir)

    ax_runner = TorchXRunner(
        tracker_base="/tmp/",
        component=trainer,
        # NOTE: To launch this job on a cluster instead of locally you can
        # specify a different scheduler and adjust args appropriately.
        scheduler="local_cwd",
        component_const_params={"log_path": log_dir},
        cfg={},
    )

    parameters = [
        # NOTE: In a real-world setting, hidden_size_1 and hidden_size_2
        # should probably be powers of 2, but in our simple example this
        # would mean that num_params can't take on that many values, which
        # in turn makes the Pareto frontier look pretty weird.
        RangeParameter(
            name="hidden_size_1",
            lower=16,
            upper=128,
            parameter_type=ParameterType.INT,
            log_scale=True,
        ),
        RangeParameter(
            name="hidden_size_2",
            lower=16,
            upper=128,
            parameter_type=ParameterType.INT,
            log_scale=True,
        ),
        RangeParameter(
            name="learning_rate",
            lower=1e-4,
            upper=1e-2,
            parameter_type=ParameterType.FLOAT,
            log_scale=True,
        ),
        RangeParameter(
            name="dropout",
            lower=0.0,
            upper=0.5,
            parameter_type=ParameterType.FLOAT,
        ),
    ]

    search_space = SearchSpace(
        parameters=parameters,
        # NOTE: In practice, it may make sense to add a constraint
        # hidden_size_2 <= hidden_size_1
        parameter_constraints=[],
    )

    class MyTensorboardMetric(TensorboardMetric):

        # NOTE: We need to tell the new Tensorboard metric how to get the id /
        # file handle for the tensorboard logs from a trial. In this case
        # our convention is to just save a separate file per trial in
        # the pre-specified log dir.
        def _get_event_multiplexer_for_trial(self, trial):
            mul = event_multiplexer.EventMultiplexer(max_reload_threads=20)
            mul.AddRunsFromDirectory(Path(log_dir).joinpath(str(trial.index)).as_posix(), None)
            mul.Reload()

            return mul

        # This indicates whether the metric is queryable while the trial is
        # still running. This is required for early stopping to monitor the
        # progress of the running trial.ArithmeticError
        @classmethod
        def is_available_while_running(cls):
            return True
        
    val_acc = MyTensorboardMetric(
        name="val_acc",
        tag="val_acc",
        lower_is_better=False,
    )

    opt_config = OptimizationConfig(
        objective=Objective(
            metric=val_acc,
            minimize=False,
        )
    )

    percentile_early_stopping_strategy = PercentileEarlyStoppingStrategy(
        # stop if in bottom 70% of runs at the same progression
        percentile_threshold=70,
        # the trial must have passed `min_progression` steps before early stopping is initiated
        # note that we are using `normalize_progressions`, so this is on a scale of [0, 1]
        min_progression=0.3,
        # there must be `min_curves` completed trials and `min_curves` trials reporting data in
        # order for early stopping to be applicable
        min_curves=5,
        # specify, e.g., [0, 1] if the first two trials should never be stopped
        trial_indices_to_ignore=None,
        normalize_progressions=True,
    )

    experiment = Experiment(
        name="torchx_mnist",
        search_space=search_space,
        optimization_config=opt_config,
        runner=ax_runner,
    )

    if SMOKE_TEST:
        total_trials = 6
    else:
        total_trials = 15  # total evaluation budget

    gs = choose_generation_strategy(
        search_space=experiment.search_space,
        optimization_config=experiment.optimization_config,
        num_trials=total_trials,
    )

    scheduler = Scheduler(
        experiment=experiment,
        generation_strategy=gs,
        options=SchedulerOptions(
            total_trials=total_trials,
            max_pending_trials=5,
            early_stopping_strategy=percentile_early_stopping_strategy,
        ),
    )

    scheduler.run_all_trials()