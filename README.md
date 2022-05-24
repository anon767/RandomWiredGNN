# Random Wired Network

Pytorch Geometric Module based on "Dont stack layers in graph neural networks, wire them randomly" by Valsesia et al. ICLR 2021

## Usage

```Python
n_layers = 5
features = 5

model = RandomWireGCN(layers=n_layers, features=features, p=0.5)
```
