import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation import ExU
from model.activation import LinReLU

"""
Code from https://github.com/lemeln/nam (Neural Additive Models:
Interpretable Machine Learning with Neural Nets)
"""


class FeatureNN(torch.nn.Module):
    """Neural Network model for each individual feature."""

    def __init__(
        self,
        input_shape: int,
        feature_num: int,
        num_units: int,
        dropout: float,
        hidden_sizes: list = [64, 32],
        activation: str = "relu",
        num_gaus: int = 2,
        lambda_var: int = 0.1,
        alpha: int = 0.1,
    ) -> None:
        """Initializes FeatureNN hyperparameters.

        Args:
          num_units: Number of hidden units in first hidden layer.
          dropout: Coefficient for dropout regularization.
          feature_num: Feature Index used for naming the hidden layers.
        """
        super(FeatureNN, self).__init__()
        self._input_shape = input_shape
        self._num_units = num_units
        self._feature_num = feature_num
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self.num_gaus = num_gaus

        all_hidden_sizes = [self._num_units] + self._hidden_sizes

        layers = []

        # self.dropout = nn.Dropout(p=dropout)

        ## First layer is ExU
        if self._activation == "exu":
            layers.append(ExU(in_features=input_shape, out_features=num_units))
        else:
            layers.append(LinReLU(in_features=input_shape, out_features=num_units))

        ## Hidden Layers
        for in_features, out_features in zip(all_hidden_sizes, all_hidden_sizes[1:]):
            layers.append(LinReLU(in_features, out_features))

        ## Last Linear Layer
        layers.append(
            nn.Linear(
                in_features=all_hidden_sizes[-1],
                out_features=self.num_gaus * 3,
                bias=False,
            )
        )

        self.model = nn.ModuleList(layers)

    def forward(self, inputs) -> torch.Tensor:
        """Computes FeatureNN output with either evaluation or training
        mode."""
        z = inputs.unsqueeze(1)
        for layer in self.model:
            # TODO: ADD dropout
            z = layer(z)

        return z
