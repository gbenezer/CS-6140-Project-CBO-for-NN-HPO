# code from https://ax.dev/docs/tutorials/visualizations/

import numpy as np
import os
from pathlib import Path
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.measurement.synthetic_functions import hartmann6
import plotly.io as pio
pio.renderers.default = "vscode"

interactive_path = Path(os.getcwd()) / "plots" / "html_plots" / "output_testing_1"
static_path = Path(os.getcwd()) / "plots" / "static_plots" / "output_testing_1"

interactive_path.mkdir(parents=True)
static_path.mkdir(parents=True)

noise_sd = 0.1
param_names = [f"x{i+1}" for i in range(6)]  # x1, x2, ..., x6


def noisy_hartmann_evaluation_function(parameterization):
    x = np.array([parameterization.get(p_name) for p_name in param_names])
    noise1, noise2 = np.random.normal(0, noise_sd, 2)

    return {
        "hartmann6": (hartmann6(x) + noise1, noise_sd),
        "l2norm": (np.sqrt((x**2).sum()) + noise2, noise_sd),
    }
    
ax_client = AxClient()
ax_client.create_experiment(
    name="test_visualizations",
    parameters=[
        {
            "name": p_name,
            "type": "range",
            "bounds": [0.0, 1.0],
        }
        for p_name in param_names
    ],
    objectives={"hartmann6": ObjectiveProperties(minimize=True)},
    outcome_constraints=["l2norm <= 1.25"],
)

for i in range(20):
    parameters, trial_index = ax_client.get_next_trial()
    # Local evaluation here can be replaced with deployment to external system.
    ax_client.complete_trial(
        trial_index=trial_index, raw_data=noisy_hartmann_evaluation_function(parameters)
    )

model = ax_client.generation_strategy.model

# extracting the dictionaries defining the interactive plots
print("making plot data")
plotly_metric_contour = interact_contour(model=model, metric_name="hartmann6").data
plotly_constraint_contour = interact_contour(model=model, metric_name="l2norm").data
plotly_tradeoff = plot_objective_vs_constraints(model, "hartmann6", rel=False).data
cv_results = cross_validate(model)
plotly_cv = interact_cross_validation(cv_results).data
plotly_tile = interact_fitted(model, rel=False).data


# saving them to html files (directory needs to be generated a priori)
pio.write_html(fig=plotly_metric_contour,
               file= interactive_path / "example_hartmann_interactive_contour.html")
pio.write_html(fig=plotly_constraint_contour,
               file= interactive_path / "example_l2norm_interactive_contour.html")
pio.write_html(fig=plotly_tradeoff,
               file=interactive_path / "example_tradeoff_interactive_plot.html")
pio.write_html(fig=plotly_cv,
               file=interactive_path / "example_cv_plot.html")
pio.write_html(fig=plotly_tile,
               file=interactive_path / "example_tile_plot.html")

# write static plot to svg (directory needs to be generated a priori)
plotly_static_contour = ax_client.get_contour_plot(param_x="x5", param_y="x6", metric_name="hartmann6").data
pio.write_image(fig=plotly_static_contour,
                file=static_path /"example_static_contour", format="svg")
