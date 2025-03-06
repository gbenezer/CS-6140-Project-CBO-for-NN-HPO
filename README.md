# Influence of Acquisition Function and Environmental Budget Variables on Constrained Bayesian Optimization of Neural Network Hyperparameters

This project aims to evaluate how the Alternating Direction of Multipliers-Bayesian Optimization (ADMMBO) technique for single-objective constrained Bayesian optimization performs on tuning neural network hyperparameters using different acquisition functions and using environmental variables (like training set size, number of epochs, or function evaluations) as tunable hyperparameters in and of themselves.

The current plan is to use the [PyTorch](https://pytorch.org/) and/or [PyTorch Lightning](https://pytorch.org/) for the implementation of the base neural network generation functionality, [Ax](https://ax.dev/) and [BoTorch](https://botorch.org/) and [GPyTorch](https://gpytorch.ai/) for Bayesian optimization using Gaussian Process-based surrogate models and implemented acquisition functions, and manual implementation of the proximal gradient method Alternating Direction Method of Multipliers (unless I can get the implementation in [PyProximal](https://pyproximal.readthedocs.io/en/stable/index.html) to work)

## General Overview of ADMMBO

For a more detailed treatment of the approach, see the paper by *Ariafar et. al.* [here](https://jmlr.org/papers/v20/18-227.html). The general approach in the application of neural network hyperparameter optimization and architecture search is as follows:

1. Define the hyparparameter space $\mathcal{S}$ and initial hyper-hyperparameters
2. Sample the space randomly to generate an initial set of hyperparameter combinations
    - Generate these combinations in a space-filling manner if possible
3. Evaluate the objective function(s) for each generated and trained neural network
4. While the primal and dual residuals are above tolerance thresholds and the budget of ADMM iterations is not exhausted
    1. Carry out Bayesian optimization on the optimality subproblem and add the gathered data to the set of evaluated hyperparameter combinations
    2. For each constraint, carry out Bayesian optimization on the feasibility subproblem and add the gathered data to the set of evaluated hyperparameter combinations