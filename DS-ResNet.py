from detectron2.layers import (CNNBottleneckBlockBase)
import torch
import torch.nn as nn

# 繼承torch.nn.Module
# super意思為子類繼承父類所有的屬性和方法
# 若沒用使用super 只有繼承到父類的方法，沒有繼承到屬性，故無法使用父類的屬性
class BottleneckBottleneckBlock(CNNBottleneckBlockBase):
    # __init__(self)內是"屬性(Attribute)"
    # 其他def的function為"方法(method)"

    def __init__(
        self,
        in_channels,
        out_channels,
        *,
        bottleneck_channels,
        stride=1,
        num_groups=1,
        norm="BN",
        stride_in_1x1=False,
        dilation=1,
    ):

        super().__init__(in_channels, out_channels, stride=1)

        if in_channels != out_channels:
            self.shortcut = Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=stride,
                bias=False,
                norm=get_norm(norm, out_channels),
            )

        else:
            self.shortcut = None

        stride_1x1, stride_3x3 = (stride, 1) if stride_in_1x1 else (1, stride)

        self.depthwise_conv1 = Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=in_channels,
            # norm=get_norm(norm, bottleneck_channels),
            # norm=in_channels,
        )
        self.pointwise_conv1 = Conv2d(
            in_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_1x1,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )

        # self.depthwise_separable_conv1 = torch.nn.Sequential(self.depthwise_conv1, self.pointwise_conv1)
        #

        self.depthwise_conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=3,
            padding=1 * dilation,
            stride=1,
            bias=False,
            groups=bottleneck_channels,
            # groups=num_groups,
            dilation=dilation,
            norm=get_norm(norm, bottleneck_channels),
        )
        self.pointwise_conv2 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=stride_3x3,
            bias=False,
            norm=get_norm(norm, bottleneck_channels),
        )
        # self.depthwise_separable_conv2 = torch.nn.Sequential(self.depthwise_conv2, self.pointwise_conv2)
        #

        self.depthwise_conv3 = Conv2d(
            bottleneck_channels,
            bottleneck_channels,
            kernel_size=1,
            stride=1,
            bias=False,
            groups=bottleneck_channels,
            norm=get_norm(norm, bottleneck_channels),
        )
        self.pointwise_conv3 = Conv2d(
            bottleneck_channels,
            out_channels,
            kernel_size=1,
            bias=False,
            norm=get_norm(norm, out_channels),
        )

        # self.depthwise_separable_conv3 = torch.nn.Sequential(self.depthwise_conv3, self.pointwise_conv3)

        for layer in [self.depthwise_conv1, self.pointwise_conv1, self.depthwise_conv2, self.pointwise_conv2, self.shortcut]:
        # for layer in [self.conv1, self.depthwise_conv2, self.pointwise_conv2, self.conv3, self.shortcut]:
        # for layer in [self.depthwise_conv1, self.pointwise_conv1, self.depthwise_conv2, self.pointwise_conv2, self.depthwise_conv3, self.pointwise_conv3, self.depthwise_shortcut, self.pointwise_shortcut, self.shortcut]:
            if layer is not None:  # shortcut can be None
                weight_init.c2_msra_fill(layer)

    def forward(self, x):
        
        out = self.depthwise_conv1(x)
        out = self.pointwise_conv1(out)
        out = F.relu_(out)

        out = self.depthwise_conv2(out)
        out = self.pointwise_conv2(out)
        out = F.relu_(out)

        out = self.depthwise_conv3(out)
        out = self.pointwise_conv3(out)

        if self.shortcut is not None:
            shortcut = self.shortcut(x)
        else:
            shortcut = x

        # x += shortcut就是element-wise add的操作
        out += shortcut
        out = F.relu_(out)
        return out
        
class ResNet(nn.Module): # [3, 4, 6, 3]
    def __init__(self, BottleneckBottleneckBlock, layers, image_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(BottleneckBlock, layers[0], intermediate_channels=64, stride=1)
        self.layer2 = self._make_layer(BottleneckBlock, layers[1], intermediate_channels=128, stride=2)
        self.layer3 = self._make_layer(BottleneckBlock, layers[2], intermediate_channels=256, stride=2)
        self.layer4 = self._make_layer(BottleneckBlock, layers[3], intermediate_channels=512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    # Either if we half the input space for ex, 56*56 -> 28*28(stride=2), or channels changes
    # we need to adapt the identity(skip connection)
    # so it will be able to be added to the layers that's ahead
    def _make_layer(self, BottleneckBlock, num_residual_BottleneckBlocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 4:
            identity_downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    intermediate_channels*4,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(intermediate_channels * 4),
            )

        # This BottleneckBlock which is the first BottleneckBlock that we append to layers
        # is gonna change the number of channels
        layers.append(BottleneckBlock(self.in_channels, intermediate_channels, identity_downsample, stride))

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels * 4

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer
        # then finally back to 256. Hence no identity downsample is needed, since=1
        # and also same amount of channels
        for i in range(num_residual_BottleneckBlocks - 1):
            layers.append(BottleneckBlock(self.in_channels, intermediate_channels)) # 256 -> 64, 64*4(256) again

        return nn.Sequential(*layers)

def ResNet50(img_channels=3, num_classes=1000):
    return ResNet(BottleneckBlock, [3,4,6,3], img_channels, num_classes)

def ResNet101(img_channels=3, num_classes=1000):
    return ResNet(BottleneckBlock, [3,4,23,3], img_channels, num_classes)

def ResNet152(img_channels=3, num_classes=1000):
    return ResNet(BottleneckBlock, [3,8,36,3], img_channels, num_classes)

def test():
    net = ResNet50() # net是RestNet50 類別(Class)的物件(object)
    x = torch.randn(2, 3, 224, 224)
    y = net(x).to('cuda')
    print(y.shape)

test()