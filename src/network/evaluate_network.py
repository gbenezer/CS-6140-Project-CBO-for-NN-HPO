# File to determine how function evaluation (neural network hyperparameter performance evaluation)
# will work

# Plans as of 09Mar2025 are to implement method for generation of test accuracy, inference time,
# and inference memory performance as three separate objective functions
# Inference time and memory utilization are subject to more noise, so that may be a consideration

# TODO: find way to extract metrics from TensorBoard and other logs
# NOTE: naming/organization of lightning log directory likely needs to be handled by Ax/BoTorch portion