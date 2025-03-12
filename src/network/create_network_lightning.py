# Functions to output custom neural networks using PyTorch Lightning
# current restrictions/assumptions are:
# - 3 layers (could be changed in theory)
# - feedforward (can't be changed easily)
# - uses only Adam optimizer (could be changed in theory)

# main import statements
import torch
from torch import nn
import lightning as L
from typing_extensions import Literal
from torchmetrics.functional.classification.accuracy import multiclass_accuracy
from torchmetrics.functional.regression.mse import mean_squared_error
from torchmetrics.functional.regression.nrmse import normalized_root_mean_squared_error

# Some code adapted from https://github.com/pytorch/tutorials/blob/main/intermediate_source/mnist_train_nas.py
# to be able to handle more general feedforward architecture for both classification and regression

# TODO: Docstring
# TODO: further commenting

def create_ff_model(
    task: Literal["regression", "classification"],
    input_shape: tuple,
    number_input_features: int,
    number_output_features: int,
    input_dropout_probability: float,
    hidden_dropout_probability: float,
    output_dropout_probability: float,
    hidden_layer_nodes_1: int,
    hidden_layer_nodes_2: int,
    hidden_layer_nodes_3: int,
    activation: nn.Module,
    loss: nn.modules.loss._Loss,
    learning_rate: float,
    beta1: float,
    beta2: float,
    w_decay: float,
):

    class FeedForwardModel(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Create a PyTorch model
            layers = [nn.Flatten(), nn.Dropout(p=input_dropout_probability)]
            width = number_input_features
            
            # following could be changed so that a list of hidden layer node numbers is passed in
            # to allow for a variable number of hidden layers
            # (and same with hidden layer dropout probabilities along with activation functions)
            
            # dimensionality of hyperparameter space would change as the number of hidden layers was varied
            # may explode
            hidden_layers = [
                hidden_layer_nodes_1,
                hidden_layer_nodes_2,
                hidden_layer_nodes_3,
            ]
            
            num_params = 0
            for hidden_size in hidden_layers:
                if hidden_size > 0:
                    layers.append(nn.Linear(width, hidden_size))
                    layers.append(activation)
                    layers.append(nn.Dropout(p=hidden_dropout_probability))
                    num_params += width * hidden_size
                    width = hidden_size
            # neccessary pop to remove the last dropoout and replace it with
            # output probability
            layers.pop()
            layers.append(nn.Dropout(p=output_dropout_probability))
            layers.append(nn.Linear(width, number_output_features))
            num_params += width * number_output_features

            # Save the model and parameter counts
            self.num_params = num_params
            self.model = nn.Sequential(
                *layers
            )  
            
            # for graph tracing
            self.example_input_array = torch.rand(size=input_shape)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            
            x, y = batch
            yhat = self(x)
            training_loss = loss(yhat, y)
            self.log("training_loss", training_loss)
            self.log(
                "training_loss_total", training_loss, reduce_fx="sum"
            )

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("training_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("training_mse", mse)
                self.log("training_nrmse_mean", nrmse_mean)
                self.log("training_nrmse_range", nrmse_range)
                self.log("training_nrmse_std", nrmse_std)

            return training_loss

        def validation_step(self, batch, batch_idx):
            
            x, y = batch
            yhat = self(x)
            validation_loss = loss(yhat, y)
            self.log("mean_validation_loss", validation_loss)
            self.log(
                "cumulative_validation_loss",
                validation_loss,
                reduce_fx="sum",
            )

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("validation_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("validation_mse", mse)
                self.log("validation_nrmse_mean", nrmse_mean)
                self.log("validation_nrmse_range", nrmse_range)
                self.log("validation_nrmse_std", nrmse_std)
            return validation_loss

        def test_step(self, batch, batch_idx):
            
            x, y = batch
            yhat = self(x)
            test_loss = loss(yhat, y)
            self.log("mean_test_loss", test_loss)
            self.log("cumulative_test_loss", test_loss, reduce_fx="sum")

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("test_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("test_mse", mse)
                self.log("test_nrmse_mean", nrmse_mean)
                self.log("test_nrmse_range", nrmse_range)
                self.log("test_nrmse_std", nrmse_std)

            return test_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=w_decay,
            )
            return optimizer

    return FeedForwardModel()

def create_ff_model_varied_layers(
    task: Literal["regression", "classification"],
    input_shape: tuple,
    number_input_features: int,
    number_output_features: int,
    input_dropout_probability: float,
    hidden_dropout_probability: float,
    output_dropout_probability: float,
    hidden_layer_nodes: tuple,
    activations: list,
    loss: nn.modules.loss._Loss,
    learning_rate: float,
    beta1: float,
    beta2: float,
    w_decay: float,
):
    
    # TODO: more explicitly restrict activations variable
    # TODO: explicitly check on number of activations versus number of layers

    class FeedForwardModel(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Create a PyTorch model
            layers = [nn.Flatten(), nn.Dropout(p=input_dropout_probability)]
            width = number_input_features
            
            num_params = 0
            for idx, hidden_size in enumerate(hidden_layer_nodes):
                if hidden_size > 0:
                    layers.append(nn.Linear(width, hidden_size))
                    layers.append(activations[idx])
                    layers.append(nn.Dropout(p=hidden_dropout_probability))
                    num_params += width * hidden_size
                    width = hidden_size
            # neccessary pop to remove the last dropoout and replace it with
            # output probability
            layers.pop()
            layers.append(nn.Dropout(p=output_dropout_probability))
            layers.append(nn.Linear(width, number_output_features))
            num_params += width * number_output_features

            # Save the model and parameter counts
            self.num_params = num_params
            self.model = nn.Sequential(
                *layers
            )  
            
            # for graph tracing
            self.example_input_array = torch.rand(size=input_shape)

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            
            x, y = batch
            yhat = self(x)
            training_loss = loss(yhat, y)
            self.log("training_loss", training_loss)
            self.log(
                "training_loss_total", training_loss, reduce_fx="sum"
            )

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("training_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("training_mse", mse)
                self.log("training_nrmse_mean", nrmse_mean)
                self.log("training_nrmse_range", nrmse_range)
                self.log("training_nrmse_std", nrmse_std)

            return training_loss

        def validation_step(self, batch, batch_idx):
            
            x, y = batch
            yhat = self(x)
            validation_loss = loss(yhat, y)
            self.log("mean_validation_loss", validation_loss)
            self.log(
                "cumulative_validation_loss",
                validation_loss,
                reduce_fx="sum",
            )

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("validation_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("validation_mse", mse)
                self.log("validation_nrmse_mean", nrmse_mean)
                self.log("validation_nrmse_range", nrmse_range)
                self.log("validation_nrmse_std", nrmse_std)
            return validation_loss

        def test_step(self, batch, batch_idx):
            
            x, y = batch
            yhat = self(x)
            test_loss = loss(yhat, y)
            self.log("mean_test_loss", test_loss)
            self.log("cumulative_test_loss", test_loss, reduce_fx="sum")

            if task == "classification":
                preds = torch.argmax(yhat, dim=1)
                acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
                self.log("test_accuracy", acc)

            elif task == "regression":
                mse = mean_squared_error(yhat, y, num_outputs=number_output_features)
                nrmse_mean = normalized_root_mean_squared_error(
                    yhat, y, normalization="mean", num_outputs=number_output_features
                )
                nrmse_range = normalized_root_mean_squared_error(
                    yhat, y, normalization="range", num_outputs=number_output_features
                )
                nrmse_std = normalized_root_mean_squared_error(
                    yhat, y, normalization="std", num_outputs=number_output_features
                )
                self.log("test_mse", mse)
                self.log("test_nrmse_mean", nrmse_mean)
                self.log("test_nrmse_range", nrmse_range)
                self.log("test_nrmse_std", nrmse_std)

            return test_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=w_decay,
            )
            return optimizer

    return FeedForwardModel()
