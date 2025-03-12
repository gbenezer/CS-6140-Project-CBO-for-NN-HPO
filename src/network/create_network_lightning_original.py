# Functions to output custom neural networks using PyTorch Lightning
# current restrictions are:
# - 3 layers (cannot vary due to hierarchical nature of hyperparameter)
# - uniform activation function across all layers
# - only two supported activation functions (Sigmoid and Relu)
# - feedforward

# TODO: finish functionality

# import statements
import torch
from torch import nn
from torch.optim import Adam
import lightning as L


def create_ff_pl_network(
    loss: nn.modules.loss._Loss,
    number_input_features: int,
    number_output_features: int,
    input_dropout_probability: float,
    hidden_dropout_probability: float,
    output_dropout_probability: float,
    hidden_layer_nodes_1: int,
    hidden_layer_nodes_2: int,
    hidden_layer_nodes_3: int,
    relu: bool,
    learning_rate: float,
    beta1: float,
    beta2: float,
    w_decay: float,
) -> L.LightningModule:

    # TODO: implement better activation function logic

    if relu:
        activation = nn.ReLU()
    else:
        activation = nn.Sigmoid()

    class customModel(L.LightningModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.flatten = nn.Flatten()
            self.input_dropout = nn.Dropout(p=input_dropout_probability)
            self.stack = nn.Sequential(
                nn.Linear(
                    in_features=number_input_features, out_features=hidden_layer_nodes_1
                ),
                activation,
                nn.Dropout(p=hidden_dropout_probability),
                nn.Linear(
                    in_features=hidden_layer_nodes_1, out_features=hidden_layer_nodes_2
                ),
                activation,
                nn.Dropout(p=hidden_dropout_probability),
                nn.Linear(
                    in_features=hidden_layer_nodes_2, out_features=hidden_layer_nodes_3
                ),
                activation,
                nn.Dropout(p=output_dropout_probability),
                nn.Linear(
                    in_features=hidden_layer_nodes_3,
                    out_features=number_output_features,
                ),
            )

        def forward(self, x):
            x = self.flatten(x)  # flatten input to 1D vector
            x = self.input_dropout(x)  # dropout random amount of input
            x = self.stack(x)
            return x

        def training_step(self, batch, batch_idx, *args, **kwargs):
            X, y = batch
            yhat = self(X)
            train_loss = loss(yhat, y)
            self.log("train_loss", train_loss)
            return train_loss

        def validation_step(self, batch, batch_idx, *args, **kwargs):
            X, y = batch
            yhat = self(X)
            val_loss = loss(yhat, y)
            self.log("val_loss", val_loss)

            # if it is a classification task with CE loss, calculate accuracy
            if loss == nn.CrossEntropyLoss():
                correct = (yhat.argmax(1) == y).sum().item()
                total = y.size(0)
                self.log(
                    "accuracy",
                    correct / total,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                )

        def configure_optimizers(self):
            return Adam(
                self.parameters(),
                lr=learning_rate,
                betas=(beta1, beta2),
                weight_decay=w_decay,
            )

    output_model = customModel()
    return output_model