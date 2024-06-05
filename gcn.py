import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool


class GCN(nn.Module):
    def __init__(self, num_features, embedding_size, num_classes):
        super().__init__()

        self.conv1 = GCNConv(num_features, embedding_size)
        self.conv2 = GCNConv(embedding_size, num_classes)
        self.tanh1 = nn.Tanh()
        self.tanh2 = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, edge_index, batch_index):
        x = self.conv1(x, edge_index)
        x = self.tanh1(x)

        x = self.conv2(x, edge_index)
        x = self.tanh2(x)

        x = global_mean_pool(x, batch_index)    # 全局平均池化
        x = self.sigmoid(x)

        return x
