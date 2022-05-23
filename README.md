# Random Wired Network

Pytorch Geometric Module based on "Dont stack layers in graph neural networks, wire them randomly" by Valsesia et al. ICLR 2021

## Usage

```Python
layers = 5
channels = 100 # I used Gated Graph Neural Networks here so |channels| >= |features|
model = RandomWireGCN(features=channels, n=layers, p=0.5)
```
