# standard imports
import torch.nn as nn
import logging
from multiprocessing import freeze_support

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# # importing the correct package defined functions
from src.network.create_network_lightning import create_ff_model
from src.network.load_data import get_MNIST_data, get_Superconductivity_data
from src.network.evaluate_network import evaluate_hyperparameters
from src.Ax_BO.experiment_definition import MNIST_status_quo_parameters, Superconductivity_status_quo_parameters

# testing Lightning framework
        
if __name__ == "__main__":
    freeze_support()

    # remove sanity check and other extraneous stout printing
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("ax.modelbridge.dispatch_utils").setLevel(logging.WARNING)
    logging.getLogger("ax.service.utils.instantiation").setLevel(logging.WARNING)
    logging.getLogger("ax.service.ax_client").setLevel(logging.WARNING)


    # Classification hyperparameter evaluation testing
    
    train_set, valid_set, testset, trainloader, validloader, testloader = (
        get_MNIST_data(
            valid_fraction=0.2,
            random_seed=42,
            n_workers=15,
            batch_n=64,
            download_data=False,
        )
    )
    
    MNIST_output_dict = (
        evaluate_hyperparameters(
            task="classification",
            train_loader=trainloader,
            valid_loader=validloader,
            test_loader=testloader,
            input_shape=(1, 28, 28),
            number_input_features = 784,
            number_output_features = 10,
            loss=nn.CrossEntropyLoss(),
            log_dir_name="hyperparam_evaluation_tests/test_1",
            num_rep=1,
            max_epochs=5,
            parameterization=MNIST_status_quo_parameters,
        )
    )
    
    print(MNIST_output_dict)

    # Regression hyperparameter evaluation testing
    (
        fulldataset,
        trainset,
        validset,
        testset,
        trainloader,
        validloader,
        testloader,
    ) = get_Superconductivity_data(
        valid_fraction=0.2,
        test_fraction=0.2,
        random_seed=0,
        n_workers=15,
        batch_n=20,
        local=False,
    )
    
    Superconductivity_output_dict = (
        evaluate_hyperparameters(
            task="regression",
            train_loader=trainloader,
            valid_loader=validloader,
            test_loader=testloader,
            input_shape=(1, 1, 81),
            number_input_features = 81,
            number_output_features = 1,
            loss=nn.HuberLoss(),
            log_dir_name="hyperparam_evaluation_tests/test_2",
            num_rep=3,
            max_epochs=5,
            parameterization=Superconductivity_status_quo_parameters,
        )
    )
    
    print(Superconductivity_output_dict)