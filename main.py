import torch
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from molecule_gnn_pytorch.gcn import GCN
import numpy as np
import matplotlib.pyplot as plt
torch.manual_seed(6)

# 丢弃那些没有分子结构的分子图
dataset = MoleculeNet(root='./', name='ClinTox', pre_filter=lambda data:data.x.numel()>0)
size = len(dataset)
num_features = dataset.num_node_features
num_classes = 1     # 只用了一个标签
embedding_size = 16

# 划分训练集和测试集
train_size = int(0.8 * size)
test_size = size-train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

# 创建模型
model = GCN(num_features, embedding_size, num_classes)
# print(model)

# 训练模型
epochs = 100
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
loss = torch.nn.BCELoss()
loss_list = []


for i in range(epochs):
    epoch_loss = 0.0
    epoch_accuracy = 0.0
    total = 0
    corr = 0

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    for batch in train_loader:
        batch_size = batch.y.shape[0]

        y = torch.Tensor()
        if batch.y.shape[1] == 2:
            y = batch.y[:, 1].reshape(-1, 1).float()
        else:
            y = batch.y.reshape(-1, 1).float()

        optimizer.zero_grad()   # 将所有参数的梯度置为0
        y_pre = model(batch.x.float(), batch.edge_index, batch.batch)

        batch_loss = loss(y_pre, y)
        epoch_loss += batch_loss

        total += batch_size

        y_pre_class = (y > 0.5).float()
        corr += (y_pre_class == y).sum().item()

        batch_loss.backward()   # 反向传播
        optimizer.step()    # 使用优化器更新参数

    # scheduler.step()

    epoch_accuracy = 1.0 * corr / total

    loss_list.append(np.squeeze(epoch_loss.detach().numpy()))

    print("[Epoch{}] | training loss:{:.5f} , training accuracy:{:.2f}%".format(i, epoch_loss, epoch_accuracy * 100))

# 绘制loss曲线
steps = [i for i in range(epochs)]

plt.title("Loss Curve")
plt.xlabel("epochs")
plt.ylabel("loss value")
plt.plot(steps, loss_list, label="train loss")

plt.savefig("images/result.png")

# plt.show()


# 测试
correct = 0
cnt = 0
for test in test_loader:
    cnt += test.y.shape[0]
    y_pre= model(test.x.float(), test.edge_index, test.batch)
    y = test.y[:, 1].reshape(-1,1).float()

    y_pre_class = (y > 0.5).float()
    correct += (y_pre_class == y).sum().item()


accuracy = 1. * correct / cnt

print("test accuracy:{}".format(accuracy))

# 保存模型
torch.save(model.state_dict(),'./molecule_gnn.pth')
