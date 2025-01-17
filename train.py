import numpy as np
import scipy.io as sio
#from osgeo import gdal
from sklearn.metrics import confusion_matrix, \
    accuracy_score, classification_report, cohen_kappa_score
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import cohen_kappa_score
from utils import applyPCA, createImageCubes, splitTrainTestSet
from DynaHybNet import DynaHybNet


# 地物类别
num_classes =9
# 用于测试样本的比例test_ratio = 0.70
test_ratio = 0.75
# test_ratio =0.95
# 每个像素周围提取 patch 的尺寸
patch_size = 25
# 使用 PCA 降维，得到主成分的数量
pca_components = 15

X = sio.loadmat('data/PaviaU.mat')['paviaU']
y = sio.loadmat('data/PaviaU_gt.mat')['paviaU_gt']



print('Hyperspectral data shape: ', X.shape)
print('Label shape: ', y.shape)

print('\n... ... PCA tranformation ... ...')
X_pca = applyPCA(X, numComponents=pca_components)
print('Data shape after PCA: ', X_pca.shape)

print('\n... ... create data cubes ... ...')
X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
print('Data cube X shape: ', X_pca.shape)
print('Data cube y shape: ', y.shape)

print('\n... ... create train & test data ... ...')
Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
print('Xtrain shape: ', Xtrain.shape)
print('Xtest  shape: ', Xtest.shape)

# 改变 Xtrain, Ytrain 的形状，以符合 pytorch 的要求
Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
print('before transpose: Xtrain shape: ', Xtrain.shape)
print('before transpose: Xtest  shape: ', Xtest.shape)

# 为了适应 pytorch 结构，数据要做 transpose
Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
Xtest = Xtest.transpose(0, 4, 3, 1, 2)
print('after transpose: Xtrain shape: ', Xtrain.shape)
print('after transpose: Xtest  shape: ', Xtest.shape)

""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


""" Testing dataset"""


class TestDS(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)

    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        # 返回文件数据的数目
        return self.len


# 创建 trainloader 和 testloader
trainset = TrainDS()
testset = TestDS()
train_loader = torch.utils.data.DataLoader(dataset=trainset,
                                           batch_size=128,
                                           shuffle=True,
                                           num_workers=2)
test_loader = torch.utils.data.DataLoader(dataset=testset,
                                          batch_size=128,
                                          shuffle=False,
                                          num_workers=2)

# 使用GPU训练，可以在菜单 "代码执行工具" -> "更改运行时类型" 里进行设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 网络放到GPU上
net = DynaHybNet(num_classes).to(device)
# net = CNN_3D(n_classes=num_classes,input_features=pca_components).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
T_max=100
scheduler = CosineAnnealingLR(optimizer, T_max=T_max)


# 开始训练
for epoch in range(100):
    total_loss = 0
    correct_train = 0
    total_train = 0
    net.train()
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        # 优化器梯度归零
        optimizer.zero_grad()
        # 正向传播 + 反向传播 + 优化
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        # 计算训练集上的准确率
        _, predicted = outputs.max(1)
        total_train += labels.size(0)
        correct_train += predicted.eq(labels).sum().item()
    # 在每个epoch结束时更新学习率
    scheduler.step()

    # 计算并打印训练指标
    train_loss_avg = total_loss / len(train_loader)
    train_accuracy = 100. * correct_train / total_train
    print('[Epoch: %d]   [Train loss avg: %.4f]   [Train Accuracy: %.2f%%]' % (epoch + 1, train_loss_avg, train_accuracy))

    correct_test = 0
    total_test = 0
    # 计算并打印测试指标
    net.eval()
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total_test += labels.size(0)
            correct_test += predicted.eq(labels).sum().item()
            for i in range(len(labels)):
                class_correct[labels[i]] += (predicted[i] == labels[i]).item()  # 更新分类正确的样本数
                class_total[labels[i]] += 1  # 更新该类别样本总数

            # 计算并输出测试指标
            test_loss += criterion(outputs, labels).item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / total
        print('[Epoch: %d]   [Test loss avg: %.4f]   [Test Accuracy: %.2f%%]' % (epoch + 1, test_loss, accuracy))
print('Finished')

# 计算并输出OA和AA
overall_accuracy = 100. * correct_test / total_test
print('Overall Accuracy: %.2f%%' % overall_accuracy)

class_accuracies = [class_correct[i] / class_total[i] for i in range(len(class_correct)) if class_total[i] > 0]
average_accuracy = 100. * sum(class_accuracies) / len(class_accuracies)
print('Average Accuracy: %.2f%%' % average_accuracy)

# 获取模型的预测结果和真实标签
predicted_labels = []
true_labels = []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

# 计算并输出Kappa
kappa_score = cohen_kappa_score(true_labels, predicted_labels)
print('Kappa Score: %.2f' % kappa_score)

# 打印每个类别的准确度
for i in range(num_classes):
    if class_total[i] > 0:
        print('Accuracy of class %d : %.2f%%' % (i, 100 * class_correct[i] / class_total[i]))
    else:
        print('Accuracy of class %d : N/A' % i)