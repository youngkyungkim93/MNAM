# MNAM (Generalizing Neural Additive Models via Statistical Multimodal Analysis)

*This repo adapted and changed from lemeln/nam.

Similar to Generalized Additive Models (GAM) and Neural Additive Models (NAM), Mulitmodal Neural Additive Models (MNAM) is an interpretable model that has a seperate neural network for each input feature. The uniquness of MNAM is that it learns relationships between features and outputs in a multimodal fashion and assigns a probability to each mode. Based on a subpopulation, MNAM will activate one or more matching modes by increasing their probability. Thus, the objective of MNAM is to learn multiple relationships and activate the right relationships by automatically identifying subpopulations of interest. Similar to how GAM and NAM have fixed relationships between features and outputs, MNAM will maintain interpretability by having multiple fixed relationships. The technique is described in [Generalizing Neural Additive Models via Statistical Multimodal Analysis](https://openreview.net/pdf?id=e4f7zawfBr).


## Example Usage

```python
from model.mnam_prob import MNAM_prob
from data.data_loader import data_loader, get_num_units

# parameter for model
weight_decay = 0.00001
lr = 0.001
activation = "relu"
output_penalty = 0.00001
dropout = 0
feature_dropout = 0
pi_loss = 0.1
sparsity = 0
n_var = 2
units_multiplier = 2

# initialize model
model = MNAM_prob(
        num_inputs=X_train.shape[1],
        num_units=get_num_units(units_multiplier, 64, X_train),
        hidden_sizes=[64, 32],
        dropout=dropout,
        feature_dropout=feature_dropout,
        weight_decay=weight_decay,
        learning_rate=lr,
        activation="relu",
        output_penalty=output_penalty,
        pi_loss=pi_loss,
        sparsity=sparsity,
        num_gaus=n_var,
    )


trainer = pl.Trainer(
    max_epochs=1000,
    accelerator="gpu",
    devices=1,
    # strategy="dp",
    enable_progress_bar=True,
)
trainer.fit(model, train_dataloaders=train_loader)

test_lost = trainer.test(model, dataloaders=test_loader, verbose=False)
```
See '''example.ipynb''' for more details.


```bibtex
@article{agarwal2021neural,
  title={Neural additive models: Interpretable machine learning with neural nets},
  author={Agarwal, Rishabh and Melnick, Levi and Frosst, Nicholas and Zhang, Xuezhou and Lengerich, Ben and Caruana, Rich and Hinton, Geoffrey E},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2021}
}
```