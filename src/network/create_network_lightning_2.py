# Functions to output custom neural networks using PyTorch Lightning
# current restrictions are:
# - 3 layers (cannot vary due to hierarchical nature of hyperparameter)
# - uniform activation function across all layers
# - feedforward

# TODO: finish functionality

# import statements
import torch
from torch import nn
import lightning as L

from torchmetrics.functional.classification.accuracy import multiclass_accuracy

# Adaptation of code from https://github.com/pytorch/tutorials/blob/main/intermediate_source/mnist_train_nas.py
# hopefully to be able to handle more general feedforward architecture for both classification and regression

def create_ff_module(
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

    class FeedForwardClassifier(L.LightningModule):
        def __init__(self):
            super().__init__()

            # Create a PyTorch model
            layers = [nn.Flatten(), nn.Dropout(p=input_dropout_probability)]
            width = number_input_features
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
            layers.append(nn.Dropout(p=output_dropout_probability))
            layers.append(nn.Linear(width, number_output_features))
            num_params += width * number_output_features

            # Save the model and parameter counts
            self.num_params = num_params
            self.model = nn.Sequential(
                *layers
            )  # No need to use Relu for the last layer

        def forward(self, x):
            return self.model(x)

        def training_step(self, batch, batch_idx):
            x, y = batch
            yhat = self(x)
            training_loss = loss(yhat, y)
            self.log("training_loss", training_loss, prog_bar=False)
            preds = torch.argmax(yhat, dim=1)
            acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
            self.log("training_accuracy", acc, prog_bar=False)
            return training_loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            yhat = self(x)
            validation_loss = loss(yhat, y)
            preds = torch.argmax(yhat, dim=1)
            self.log("validation_loss", validation_loss, prog_bar=False)
            acc = multiclass_accuracy(preds, y, num_classes=number_output_features)
            self.log("validation_accuracy", acc, prog_bar=False)
            return validation_loss

        def configure_optimizers(self):
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=w_decay,
            )
            return optimizer

    return FeedForwardClassifier()
