import torch
from gcn import GCN
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader

dataset = MoleculeNet(root='./', name='ClinTox', pre_filter=lambda data:data.x.numel()>0, force_reload=True)
size = len(dataset)
num_features = dataset.num_node_features
num_classes = 1
embedding_size = 16

data_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 加载已经训练好的模型
model = GCN(num_features, embedding_size, num_classes)
model.load_state_dict(torch.load("./molecule_gnn.pth"))


correct = 0
cnt = 0
for test in data_loader:
    cnt += test.y.shape[0]
    y_pre = model(test.x.float(), test.edge_index, test.batch)
    y = test.y[:, 1].reshape(-1, 1).float()

    y_pre_class = (y > 0.5).float()
    correct += (y_pre_class == y).sum().item()

accuracy = 1. * correct / cnt

print("accuracy:{}".format(accuracy))


