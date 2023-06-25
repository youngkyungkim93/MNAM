from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.featurenn_mnam import FeatureNN
import pytorch_lightning as pl
from torch import linalg as LA
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch.distributions import Categorical


class MNAM(pl.LightningModule):

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
        num_gaus: int,
        pi_loss: float,
        sparsity: float
    ) -> None:
        super(MNAM, self).__init__()
        assert len(num_units) == num_inputs
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
        self.num_gaus = num_gaus
        self.pi_loss = pi_loss
        self.sparsity = sparsity

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList([
            FeatureNN(
                input_shape=1, 
                num_units=self.num_units[i], 
                dropout=self.dropout, feature_num=i, 
                hidden_sizes=self.hidden_sizes,
                activation = activation,
                num_gaus = self.num_gaus
            )
            for i in range(num_inputs)
        ])

        self._bias = torch.nn.Parameter(data=torch.zeros(1,num_gaus))

        # set loss function based on problem type
        self.loss_func = nn.MSELoss(reduction = "none")
        self.loss_func_test = nn.L1Loss(reduction = "none")

        # layer for computing pis
        self.pi_layer = nn.Linear(num_inputs*num_gaus, num_gaus)
        self.eval_loss_cent = nn.CrossEntropyLoss(reduction = "none")

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        list_z,list_z_pi= [], []
        loss_eval = 0
        for i in range(self.num_inputs):
            z = self.feature_nns[i](inputs[:, i])
            list_z.append(z[:,: self.num_gaus])
            list_z_pi.append(z[:,self.num_gaus :])
        return torch.cat(list_z,dim = 0), torch.cat(list_z_pi,dim = 1)

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z, z_pi = self.calc_outputs(inputs)
        # pass all the values and concat predictions
        y_hat_list = []
        for i in range(self.num_gaus):
            z_ = torch.cat(torch.split(z[:,i].view(-1,1), int(len(z)/self.num_inputs)),axis=1)
            dropout_out = self.dropout_layer(z_)
            y_hat = torch.sum(dropout_out, dim=-1) + self._bias[:,i]
            y_hat_list.append(y_hat.view(-1,1))
        
        y_hat = torch.cat(y_hat_list,dim = 1)
        
        # estimate pi hat and compute cross entropy loss
        pi_hat = self.pi_layer(z_pi)
        
        return y_hat, pi_hat, z

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        loss = self.compute_loss(x, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss = self.compute_loss(x, y)
        self.log('val_loss', loss)
        return loss
    
    def compute_loss(self, x, y):
        z, z_pi = self.calc_outputs(x)
        
        # pass all the values to evaluate which gaussians has the best performance
        loss_eval_list = []
        dropout_list = []
        for i in range(self.num_gaus):
            z_ = torch.cat(torch.split(z[:,i].view(-1,1), int(len(z)/self.num_inputs)),axis=1)
            dropout_out = self.dropout_layer(z_)
            y_hat = torch.sum(dropout_out, dim=-1) + self._bias[:,i]
            loss_eval = self.loss_func(y_hat, y)
            loss_eval_list.append(loss_eval.view(-1,1))
            dropout_list.append(torch.transpose(dropout_out,0,1).flatten().view(-1,1))
            
        loss_eval = torch.cat(loss_eval_list,dim = 1)
        dropout = torch.cat(dropout_list,dim = 1)
        loss, min_index = torch.min(loss_eval,1)
        
        # estimate pi hat and compute cross entropy loss
        pi_hat = self.pi_layer(z_pi)
        loss_cent = self.eval_loss_cent(pi_hat, min_index)

        loss += loss_cent * self.pi_loss

        # compute sparsity loss:
        if self.sparsity > 0:
            pi_hat_prob = F.softmax(pi_hat)
            # compute loss for sparsity
            loss_sp = Categorical(probs = pi_hat_prob).entropy()
            loss += (loss_sp * self.sparsity)

        # compute weight decay loss
        if self.weight_decay > 0:
            num_networks = len(self.feature_nns)
            l2_losses = [(x**2).sum() for x in self.parameters()]
            loss += (sum(l2_losses) / num_networks) * self.weight_decay
        min_index = min_index.repeat(self.num_inputs)
        
        # compute output penalty
        feature_output = dropout.gather(1, min_index.view(-1,1))
        if self.output_penalty > 0:
            loss += torch.sum(feature_output ** 2) / (feature_output.size()[0]\
            *feature_output.size()[1]) * self.output_penalty
        
        return torch.mean(loss)
    
    def compute_test_error_reg(self, x, y):
        z, z_pi = self.calc_outputs(x)
        
        # pass all the values to evaluate which gaussians has the best performance
        loss_eval_list = []
        for i in range(self.num_gaus):
            z_ = torch.cat(torch.split(z[:,i].view(-1,1), int(len(z)/self.num_inputs)),axis=1)
            dropout_out = self.dropout_layer(z_)
            y_hat = torch.sum(dropout_out, dim=-1) + self._bias[:,i]
            loss_eval = self.loss_func_test(y_hat, y)
            loss_eval_list.append(loss_eval.view(-1,1))
            
        loss_eval = torch.cat(loss_eval_list,dim = 1)
        pi_hat = self.pi_layer(z_pi)
        loss = torch.sum(loss_eval*F.softmax(pi_hat), dim = 1)
        loss, min_index = torch.min(loss_eval,1)
        return torch.mean(loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        loss_test = self.compute_test_error_reg(x, y)
        self.log('test_loss', loss_test)
        return loss_test
    

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat, pi_hat, z = self.forward(x)
        return y_hat, pi_hat, z
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,\
        gamma=0.995,step_size=1)
        return [optimizer], [scheduler]
    
