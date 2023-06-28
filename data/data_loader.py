import torch
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Union
from numpy.typing import ArrayLike
import pandas as pd
import numpy as np


def data_loader(X_train, y_train, X_val, y_val, X_test, y_test):
    # Set dataset into tensor for computation
    tensor_x_train = torch.Tensor(X_train)
    tensor_y_train = torch.Tensor(y_train)

    tensor_x_val = torch.Tensor(X_val)
    tensor_y_val = torch.Tensor(y_val)

    tensor_x_test = torch.Tensor(X_test)
    tensor_y_test = torch.Tensor(y_test)

    my_dataset_train = TensorDataset(
        tensor_x_train, tensor_y_train
    )  # create your datset
    my_dataset_val = TensorDataset(tensor_x_val, tensor_y_val)
    my_dataset_test = TensorDataset(tensor_x_test, tensor_y_test)  # create your datset
    train_loader = DataLoader(
        my_dataset_train, batch_size=len(tensor_x_test), shuffle=True, num_workers=3
    )
    val_loader = DataLoader(
        my_dataset_val, batch_size=len(tensor_x_test), shuffle=True, num_workers=3
    )
    test_loader = DataLoader(
        my_dataset_test, batch_size=len(tensor_y_test), shuffle=False, num_workers=3
    )
    return train_loader, val_loader, test_loader


def get_num_units(
    units_multiplier: int, num_basis_functions: int, X: Union[ArrayLike, pd.DataFrame]
) -> List:
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
    num_unique_vals = [len(np.unique(X[:, i])) for i in range(X.shape[1])]
    num_units = [
        min(num_basis_functions, i * units_multiplier) for i in num_unique_vals
    ]

    return num_units
