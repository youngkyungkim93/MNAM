from typing import Sequence
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.featurenn_mnam_gau import FeatureNN

import pytorch_lightning as pl
from torch import linalg as LA
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import roc_auc_score
from torch.distributions import Categorical

import numpy as np
from scipy import stats


class MNAM_prob(pl.LightningModule):
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
        num_gaus: int,
        pi_loss: float,
        sparsity: float,
    ) -> None:
        super(MNAM_prob, self).__init__()
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
        self.num_gaus = num_gaus
        self.pi_loss = pi_loss
        self.sparsity = sparsity

        self.dropout_layer = nn.Dropout(p=self.feature_dropout)

        ## Builds the FeatureNNs on the first call.
        self.feature_nns = nn.ModuleList(
            [
                FeatureNN(
                    input_shape=1,
                    num_units=self.num_units[i],
                    dropout=self.dropout,
                    feature_num=i,
                    hidden_sizes=self.hidden_sizes,
                    activation=activation,
                    num_gaus=self.num_gaus,
                )
                for i in range(num_inputs)
            ]
        )

        self._bias = torch.nn.Parameter(data=torch.zeros(2, num_gaus))

        # loss fuction
        self.loss_func = nn.GaussianNLLLoss(reduction="none")
        self.loss_func_test = nn.L1Loss(reduction="none")

        # layer for computing loss for pis
        self.pi_layer = nn.Linear(num_inputs * num_gaus, num_gaus)
        self.eval_loss_cent = nn.CrossEntropyLoss(reduction="none")
        self.var_act = nn.ELU()

    def calc_outputs(self, inputs: torch.Tensor) -> Sequence[torch.Tensor]:
        """Returns the output computed by each feature net."""
        list_mu, list_sigma, list_z_pi = [], [], []
        loss_eval = 0
        for i in range(self.num_inputs):
            z = self.feature_nns[i](inputs[:, i])
            list_mu.append(z[:, : self.num_gaus])
            list_sigma.append(z[:, self.num_gaus : self.num_gaus * 2])
            list_z_pi.append(z[:, self.num_gaus * 2 :])
        return (
            torch.cat(list_mu, dim=0),
            torch.cat(list_sigma, dim=0),
            torch.cat(list_z_pi, dim=1),
        )

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z_mu, z_sigma, z_pi = self.calc_outputs(inputs)
        # pass all the values and concat predictions
        mu_list = []
        sigma_list = []
        for i in range(self.num_gaus):
            z_mu_ = torch.cat(
                torch.split(z_mu[:, i].view(-1, 1), int(len(z_mu) / self.num_inputs)),
                axis=1,
            )
            z_sigma_ = torch.cat(
                torch.split(
                    z_sigma[:, i].view(-1, 1), int(len(z_sigma) / self.num_inputs)
                ),
                axis=1,
            )
            dropout_out_mu = self.dropout_layer(z_mu_)
            dropout_out_sigma = self.var_act(self.dropout_layer(z_sigma_)) + 1
            mu = torch.sum(dropout_out_mu, dim=-1) + self._bias[0, i]
            sigma = torch.sum(dropout_out_sigma, dim=-1)
            mu_list.append(mu.view(-1, 1))
            sigma_list.append(sigma.view(-1, 1))

        mu = torch.cat(mu_list, dim=1)
        sigma = torch.cat(sigma_list, dim=1)

        # estimate pi hat and compute cross entropy loss
        pi_hat = self.pi_layer(z_pi)

        return mu, sigma, pi_hat, z_mu, z_sigma

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        self.forward(x)
        loss = self.compute_loss(x, y)
        self.log("train_loss", loss)
        self.log("epoch", self.current_epoch)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        loss = self.compute_loss(x, y)
        self.log("val_loss", loss)
        return loss

    def compute_loss(self, x, y):
        z_mu, z_sigma, z_pi = self.calc_outputs(x)

        # pass all the values to evaluate which gaussians has the best performance
        loss_eval_list = []
        dropout_list_mu = []
        dropout_list_sigma = []
        for i in range(self.num_gaus):
            z_mu_ = torch.cat(
                torch.split(z_mu[:, i].view(-1, 1), int(len(z_mu) / self.num_inputs)),
                axis=1,
            )
            z_sigma_ = torch.cat(
                torch.split(
                    z_sigma[:, i].view(-1, 1), int(len(z_sigma) / self.num_inputs)
                ),
                axis=1,
            )
            dropout_out_mu = self.dropout_layer(z_mu_)
            dropout_out_sigma = self.var_act(self.dropout_layer(z_sigma_)) + 1
            mu = torch.sum(dropout_out_mu, dim=-1) + self._bias[0, i]
            sigma = torch.sum(dropout_out_sigma, dim=-1)
            loss_eval = self.loss_func(mu, y, sigma)
            loss_eval_list.append(loss_eval.view(-1, 1))
            dropout_list_mu.append(
                torch.transpose(dropout_out_mu, 0, 1).flatten().view(-1, 1)
            )
            dropout_list_sigma.append(
                torch.transpose(dropout_out_sigma, 0, 1).flatten().view(-1, 1)
            )

        loss_eval = torch.cat(loss_eval_list, dim=1)
        dropout_mu = torch.cat(dropout_list_mu, dim=1)
        dropout_sigma = torch.cat(dropout_list_sigma, dim=1)
        loss, min_index = torch.min(loss_eval, 1)

        # estimate pi hat and compute cross entropy loss
        pi_hat = self.pi_layer(z_pi)
        loss_cent = self.eval_loss_cent(pi_hat, min_index)

        loss += loss_cent * self.pi_loss

        # compute sparsity loss:
        if self.sparsity > 0:
            pi_hat_prob = F.softmax(pi_hat)
            # compute loss for sparsity
            loss_sp = Categorical(probs=pi_hat_prob).entropy()
            loss += loss_sp * self.sparsity

        # compute weight decay loss
        if self.weight_decay > 0:
            num_networks = len(self.feature_nns)
            l2_losses = [(x ** 2).sum() for x in self.parameters()]
            loss += (sum(l2_losses) / num_networks) * self.weight_decay
        min_index = min_index.repeat(self.num_inputs)

        # compute output penalty
        feature_output_mu = dropout_mu.gather(1, min_index.view(-1, 1))
        feature_output_sigma = dropout_sigma.gather(1, min_index.view(-1, 1))
        if self.output_penalty > 0:
            loss += (
                torch.sum(feature_output_mu ** 2)
                / (feature_output_mu.size()[0] * feature_output_mu.size()[1])
                * self.output_penalty
            )
            loss += (
                torch.sum(feature_output_sigma ** 2)
                / (feature_output_sigma.size()[0] * feature_output_sigma.size()[1])
                * self.output_penalty
            )

        return torch.mean(loss)

    def compute_test_error_reg(self, x, y):
        z_mu, z_sigma, z_pi = self.calc_outputs(x)

        # pass all the values to evaluate which gaussians has the best performance
        loss_eval_list = []
        mu_list = []
        sigma_list = []
        for i in range(self.num_gaus):
            z_mu_ = torch.cat(
                torch.split(z_mu[:, i].view(-1, 1), int(len(z_mu) / self.num_inputs)),
                axis=1,
            )
            z_sigma_ = torch.cat(
                torch.split(
                    z_sigma[:, i].view(-1, 1), int(len(z_sigma) / self.num_inputs)
                ),
                axis=1,
            )
            dropout_out_mu = self.dropout_layer(z_mu_)
            dropout_out_sigma = self.var_act(self.dropout_layer(z_sigma_)) + 1
            mu = torch.sum(dropout_out_mu, dim=-1) + self._bias[0, i]
            sigma = torch.sum(dropout_out_sigma, dim=-1)
            mu_list.append(mu.view(-1, 1))
            sigma_list.append(sigma.view(-1, 1))

            loss_eval = self.loss_func_test(mu, y)
            # loss_eval = self.loss_func(mu, y, sigma)
            loss_eval_list.append(loss_eval.view(-1, 1))

        loss_eval = torch.cat(loss_eval_list, dim=1)

        pi_hat = self.pi_layer(z_pi)
        loss = torch.sum(loss_eval * F.softmax(pi_hat), dim=1)
        mu = torch.cat(mu_list, dim=1)
        sigma = torch.cat(sigma_list, dim=1)
        num_sample = 100
        ypred = []
        for pis, mu, sig, y_t in zip(
            F.softmax(pi_hat).detach().cpu().numpy(),
            mu.detach().cpu().numpy(),
            sigma.detach().cpu().numpy(),
            y.detach().cpu().numpy(),
        ):
            # y_test_ts.append([y_t] * num_sample)
            comp_sam = np.random.choice(self.num_gaus, num_sample, p=pis)
            y_pre = []
            for j in range(self.num_gaus):
                sample_num = sum(num == j for num in comp_sam)
                y_sample = np.random.normal(mu[j], (sig[j] ** 0.5), size=sample_num)
                y_pre = np.append(y_pre, y_sample)
            ypred.append(y_pre)

        # loss, min_index = torch.min(loss_eval,1)
        return (
            torch.mean(loss),
            stats.wasserstein_distance(y.detach().cpu().numpy(), np.concatenate(ypred)),
        )

    def test_step(self, batch, batch_idx):
        x, y = batch
        mae, emd = self.compute_test_error_reg(x, y)
        self.log("test_loss_mae", mae)
        self.log("test_loss_emd", emd)
        return mae, emd

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        mu, sigma, pi_hat, z_mu, z_sigma = self.forward(x)
        return mu, sigma, pi_hat, z_mu, z_sigma

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.995, step_size=1)
        return [optimizer], [scheduler]
