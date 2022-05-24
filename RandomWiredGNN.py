import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import global_max_pool, GatedGraphConv


class RandomWireGCN(torch.nn.Module):
    def __init__(self, layers: ModuleList, hidden_size: int, p: float):
        super(RandomWireGCN, self).__init__()
        self.layers = layers
        self.hidden_size = hidden_size
        self.p = p

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        cached = [torch.zeros(len(x), self.hidden_size)] * len(self.layers)
        for i, layer in enumerate(self.layers):
            if i == 0:
                layer_result = x
            else:
                layer_result = torch.zeros(len(x), self.hidden_size)
                for k in range(i):
                    if np.random.choice([0, 1], p=[1 - self.p, self.p]) == 1:
                        layer_result += cached[k] / i
            cached[i] = self.layers[i](layer_result, edge_index).relu()

        return cached[-1]
