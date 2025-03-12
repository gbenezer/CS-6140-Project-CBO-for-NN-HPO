# File to determine how function evaluation (neural network hyperparameter performance evaluation)
# will work

# Plans as of 09Mar2025 are to implement method for generation of test accuracy, inference time,
# and inference memory performance as three separate objective functions
# Inference time and memory utilization are subject to more noise, so that may be a consideration

# TODO: find way to extract metrics from TensorBoard and other logs
# NOTE: naming/organization of lightning log directory likely needs to be handled by Ax/BoTorch portion

from pathlib import Path
import logging
import time
import os
import pandas as pd
import torch
import torch.nn as nn
import lightning as L
from lightning.pytorch import loggers as pl_loggers
from multiprocessing import freeze_support
from lightning.pytorch.profilers import SimpleProfiler
from lightning.pytorch.callbacks import ModelSummary
from load_data import get_MNIST_data
from create_network_lightning import create_ff_model


if __name__ == "__main__":
    freeze_support()

    # remove sanity check and other extraneous stout printing
    logging.getLogger("lightning").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
    logging.getLogger("")

    for i in range(1):

        train_set, valid_set, testset, trainloader, validloader, testloader = (
            get_MNIST_data(
                valid_fraction=0.2,
                random_seed=42,
                n_workers=15,
                batch_n=64,
                download_data=False,
            )
        )

        test_lighting_module = create_ff_model(
            task="classification",
            input_shape=(1, 28, 28),
            number_input_features=784,
            number_output_features=10,
            input_dropout_probability=0.2,
            hidden_dropout_probability=0.5,
            output_dropout_probability=0.1,
            hidden_layer_nodes_1=1000,
            hidden_layer_nodes_2=200,
            hidden_layer_nodes_3=50,
            activation=nn.ReLU(),
            loss=nn.CrossEntropyLoss(),
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            w_decay=0,
        )
        
        print("params:", test_lighting_module.num_params)
        print("model:", test_lighting_module.model)

        # may want to functionalize the construction of a Trainer and loggers if not explicit in future
        # evaluate_network function

        tensorboard_logger = pl_loggers.TensorBoardLogger(save_dir="tb_logs/", name="logging_tests", log_graph=True)
        csv_logger = pl_loggers.CSVLogger(save_dir="csv_logs/", name="logging_tests")
        tensorboard_dir_path = Path(tensorboard_logger.log_dir)
        checkpoint_dir_path = tensorboard_dir_path / "checkpoints"
        csv_dir_path = Path(csv_logger.log_dir)

        simple_profiler = SimpleProfiler(filename="profiling_output")

        trainer = L.Trainer(
            max_epochs=5,
            enable_progress_bar=True,
            logger=[tensorboard_logger, csv_logger],
            profiler=simple_profiler,
            callbacks=[ModelSummary(max_depth=2)]
        )
        
        fit_start=time.time()

        trainer.fit(
            model=test_lighting_module,
            train_dataloaders=trainloader,
            val_dataloaders=validloader,
        )
        
        fit_end = time.time()
        print("training time:", (fit_end - fit_start) / 60.0)
        
        test_start = time.time()

        trainer.test(model=test_lighting_module, dataloaders=testloader)
        
        test_end = time.time()
        
        print("testing time: ", (test_end - test_start) / 60.0)
        
        size = 0
        for file in os.scandir(checkpoint_dir_path):
            size += os.path.getsize(file)
            
        print("checkpoint size:", size, "bytes")
