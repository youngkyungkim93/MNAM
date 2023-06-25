from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.featurenn import FeatureNN
import pytorch_lightning as pl
from torch import linalg as LA
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score


class NAM(pl.LightningModule):

    def __init__(
        self,
        num_inputs: int,
        num_units: list,
        hidden_sizes: list,
        dropout: float,
        feature_dropout: float,
        output_penalty: float,
        weight_decay: float,
        learning_rate: float,
        activation: str,
        problem_type: str,
    ) -> None:
        super(NAM, self).__init__()
        # assert len(num_units) == num_inputs
        # hyperparameters  for the model
        self.num_inputs = num_inputs
        self.num_units = num_units
        self.hidden_sizes = hidden_sizes
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.output_penalty = output_penalty
        self.weight_decay = weight_decay
        self.lr = learning_rate
        self.activation = activation
        self.problem_type = problem_type

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=self.num_units[i], 
                dropout=self.dropout, feature_num=i, 
                hidden_sizes=self.hidden_sizes,
                activation = activation
            )
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1))

        # set loss function based on problem type
        if problem_type == "regression":
            self.loss_func = nn.MSELoss()
        
        else:
            self.loss_func = nn.BCEWithLogitsLoss()

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        return [self.feature_nns[i](inputs[:, i]) for i in range(self.num_inputs)]

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        individual_outputs = self.calc_outputs(inputs)
        conc_out = torch.cat(individual_outputs, dim=-1)
        dropout_out = self.dropout_layer(conc_out)

        out = torch.sum(dropout_out, dim=-1)
        return out + self._bias, dropout_out

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        output, feature_output= self.forward(x)
        loss = self.compute_loss(output, y, feature_output)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        output, feature_output= self.forward(x)
        loss = self.compute_loss(output, y, feature_output)
        self.log('val_loss', loss)
        return loss
    
    def compute_loss(self, pred, y, feature_output):
        # compute mse loss
        loss = self.loss_func(pred, y)

        # compute weight decay loss
        if self.weight_decay > 0:
            num_networks = len(self.feature_nns)
            l2_losses = [(x**2).sum() for x in self.parameters()]
            loss += ((sum(l2_losses) / num_networks) * self.weight_decay)
        
        # compute output penalty
        if self.output_penalty > 0:
            loss += (torch.sum(feature_output ** 2) / (feature_output.size()[0]\
            *feature_output.size()[1]) * self.output_penalty)
            # loss += (LA.norm(feature_output)/ (feature_output.size()[0]\
            # *feature_output.size()[1])) * self.output_penalty
        
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        output, dropout_out = self.forward(x)
        loss = self.loss_func(output, y)
        if self.problem_type == "regression":
            loss_test = torch.sqrt(loss)
            self.log('test_loss', loss_test)
        
        else:
            output = torch.sigmoid(output)
            output = output.detach().cpu().numpy()
            y = y.detach().cpu().numpy()
            loss_test = roc_auc_score(y, output)
            self.log('test_loss', loss_test)

        return loss_test

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,\
        gamma=0.995,step_size=1)
        return [optimizer], [scheduler]
    
