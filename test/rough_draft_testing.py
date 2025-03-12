# standard imports
import torch
import torch.nn as nn
import lightning as L
from multiprocessing import Process, freeze_support
from torch.profiler import profile, ProfilerActivity
from lightning.pytorch.profilers import (
    SimpleProfiler,
    AdvancedProfiler,
    PyTorchProfiler,
)

# reducing some of the precision to hopefully increase performance
torch.set_float32_matmul_precision("high")

# adding all the modules and submodules to the path
import sys

sys.path.insert(0, "C:/Users/Gil/Documents/Repositories/Python/CS_6140/Project")

# defining the device
dev = "cuda"

# # importing the correct package defined functions
from src.network.create_network_lightning import create_ff_model
from src.network.load_data import get_MNIST_data


# portions of code that need to be modularized in the network subpackage

# # profiler testing (needs modularization)
# with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, profile_memory=True) as prof:
#     test_loss, test_accuracy, average_test_loss = test_network(dataloader=testloader,
#                                                                model=testing_network,
#                                                                loss_fn=criterion,
#                                                                device=dev)

# for fEventAvg in prof.key_averages():
#     if fEventAvg.key != "[memory]":
#         print("Total CPU Time for", fEventAvg.key, ":", fEventAvg.cpu_time_total, sep=" ")
#         print("Total CUDA Time for", fEventAvg.key, ":", fEventAvg.device_time_total, sep=" ")
#         print("Total CPU Memory for", fEventAvg.key, ":", fEventAvg.cpu_memory_usage, sep=" ")
#         print("Total CUDA Memory for", fEventAvg.key, ":", fEventAvg.device_memory_usage, sep=" ")

# print(prof.key_averages().table())

# found method to get memory and time used during inference, so likely will try for a three-objective optimization
# overhead is high, so may not be effective for training

# # testing Lightning framework

if __name__ == "__main__":
    freeze_support()

    import logging

    logging.getLogger("lightning").setLevel(logging.ERROR)

    # configure logging at the root level of Lightning
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    for i in range(4):

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
            number_input_features=784,
            number_output_features=10,
            input_dropout_probability=0.2,
            hidden_dropout_probability=0.5,
            output_dropout_probability=0.1,
            hidden_layer_nodes_1=500,
            hidden_layer_nodes_2=100,
            hidden_layer_nodes_3=50,
            activation=nn.ReLU(),
            loss=nn.CrossEntropyLoss(),
            learning_rate=1e-3,
            beta1=0.9,
            beta2=0.999,
            w_decay=0,
        )

        simple_profiler = SimpleProfiler(filename="fit_profiling_output")
        trainer = L.Trainer(
            max_epochs=20,
            enable_progress_bar=True,
            logger=True,
            profiler=simple_profiler,
        )

        trainer.fit(
            model=test_lighting_module,
            train_dataloaders=trainloader,
            val_dataloaders=validloader,
        )

        trainer.test(model=test_lighting_module, dataloaders=testloader)
