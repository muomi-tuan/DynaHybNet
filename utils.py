import numpy as np

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split


# 对高光谱数据 X 应用 PCA 变换，用于降低光谱波段的维度
#pca降维：PCA降维实际上是找到原始数据中变化最大的方向，并将数据投影到这些方向上。
def applyPCA(X, numComponents):
    newX = np.reshape(X, (-1, X.shape[2]))      #将三维的高光谱数据 X 重新构造为一个二维数组，其中每一行代表一个像素，每一列代表一个光谱波段。
    pca = PCA(n_components=numComponents, whiten=True)      #创建了一个PCA对象，指定了降维后的维度为numComponents。whiten=True表示将数据白化，即对数据进行归一化处理。
    newX = pca.fit_transform(newX)      #使用PCA对象对二维数组进行降维变换，得到降维后的数据。
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))        #将降维后的数据重新塑造为原始的三维形状，其中numComponents代表新的光谱波段数。
    return newX

# 对单个像素周围提取图像块（patch）时，边缘像素就无法取了，因此，给这部分像素进行 padding 操作
def padWithZeros(X, margin=2):
    newX = np.zeros(
        (X.shape[0] + 2 * margin, X.shape[1] + 2 * margin, X.shape[2]))         # 创建一个新的零矩阵，其形状比原始数据在两个维度上分别增加了2倍的margin
    x_offset = margin
    y_offset = margin       # 设置 x 和 y 方向的偏移量，即将原始数据嵌入新矩阵的位置
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X     # 将原始数据复制到新矩阵中，实现边缘填充
    return newX


# 在每个像素周围提取 patch ，然后创建成符合 pytorch 处理的格式，目的是从一个图像张量 X 中提取出所有可能的小块（patch），并将它们和对应的标签 y 一起返回。
def createImageCubes(X, y, windowSize=9, removeZeroLabels=True):        #将window从5改为9
    # 给 X 做 padding
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros(
        (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1,
                                c - margin:c + margin + 1]
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r - margin, c - margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels > 0, :, :, :]
        patchesLabels = patchesLabels[patchesLabels > 0]
        patchesLabels -= 1
    return patchesData, patchesLabels

# # 在每个像素周围提取 patch ，然后创建成符合 pytorch 处理的格式，目的是从一个图像张量 X 中提取出所有可能的小块（patch），并将它们和对应的标签 y 一起返回。
# def createImageCubes(X, y, windowSize=5, removeZeroLabels=True):
#     result = np.zeros((1, 25, 25, 30))
#     # 给 X 做 padding
#     margin = int((windowSize - 1) / 2)
#     zeroPaddedX = padWithZeros(X, margin=margin)
#     print('*********************')
#     print(zeroPaddedX.shape[0])
#     print(margin)
#     print(zeroPaddedX.shape[1])
#     # split patches
#     # patchesData = np.zeros(
#     #     (X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
#     patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
#     patchIndex = 0
#
#     for r in range(margin, zeroPaddedX.shape[0] - margin):
#         for c in range(margin, zeroPaddedX.shape[1] - margin):
#             patch = zeroPaddedX[r - margin:r + margin + 1,
#                                 c - margin:c + margin + 1]
#             patch = np.expand_dims(patch,0)
#             # print(patch.shape)
#             # patchesData[patchIndex, :, :, :] = patch
#             patchesLabels[patchIndex] = y[r - margin, c - margin]
#             patchIndex = patchIndex + 1
#             result = np.concatenate((patch, result), 0)
#     result = result[1:,:,:,:]
#     if removeZeroLabels:
#         patchesData = result[patchesLabels > 0, :, :, :]
#         patchesLabels = patchesLabels[patchesLabels > 0]
#         patchesLabels -= 1
#     return patchesData, patchesLabels



def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=testRatio, random_state=randomState, stratify=y)
    return X_train, X_test, y_train, y_test
