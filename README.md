# Random Wired Network

Pytorch Geometric Module based on "Dont stack layers in graph neural networks, wire them randomly" by Valsesia et al. ICLR 2021

## Usage

```Python
n_layers = 5
hidden_size = 32
layers = torch.nn.ModuleList()
for _ in range(n_layers):
    layers.append(GCNConv(hidden_size, hidden_size))
model = RandomWireGCN(layers=layers, hidden_size=hidden_size, p=0.5)
```
