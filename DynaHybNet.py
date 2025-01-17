import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torch.cuda as cuda
from thop import profile

class attention3d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature):
        super(attention3d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv3d(in_planes, hidden_planes, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(hidden_planes)
        self.fc2 = nn.Conv3d(hidden_planes, K, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(K)
        self.temperature = temperature

    def updata_temperature(self):
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)

class Dynamic_conv3d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34):
        super(Dynamic_conv3d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention3d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None


        #TODO 初始化
        # nn.init.kaiming_uniform_(self.weight, )

    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):#将batch视作维度变量，进行组卷积，因为组卷积的权重是不同的，动态卷积的权重也是不同的
        softmax_attention = self.attention(x)
        batch_size, in_planes, depth, height, width = x.size()
        x = x.view(1, -1, depth, height, width)# 变化成一个维度进行组卷积
        weight = self.weight.view(self.K, -1)

        # 动态卷积的权重的生成， 生成的是batch_size个卷积参数（每个参数不同）
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size, self.kernel_size)
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv3d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv3d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-3), output.size(-2), output.size(-1))
        return output

class DynaHybNet(nn.Module):
    def __init__(self, num_classes, in_channels=1):
        super(DynaHybNet, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.dynamic_conv3d = Dynamic_conv3d(in_planes=32, out_planes=32, kernel_size=3, ratio=0.25, stride=1,
                                             padding=0, dilation=1, groups=1, bias=True, K=4, temperature=34)

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),        #从512改为了32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 15 * 15, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = self.dynamic_conv3d(x)
        x = x.view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], x.size()[4])
        x = self.conv2d_features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    x = torch.randn(4, 1, 30, 25, 25, device='cpu')  # 表示有4个样本，每个样本有1个通道，每个通道有30个光谱带，每个光谱带有25*25个像素。
    x = x.cuda()
    model = DynaHybNet(5, 1)
    model.cuda()
    model.eval()
    summary(model, (1, 30, 25, 25), device='cuda')
    with torch.no_grad():
        out = model(x)
        print(out.shape)

    print('GPU memory usage:', round(cuda.memory_allocated() / (1024 ** 2), 2), 'MB')
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total trainable parameters:', total_params)
    flops, params = profile(model, inputs=(x,))
    print('FLOPs:', flops, 'params:', params)