import torch
import torch.nn as nn
import torch.nn.functional as F


class BottleNeck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):

        super(BottleNeck, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample


    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    

class ResNet50(nn.Module):

    def __init__(self, block, layers, num_classes=1000):
        super(ResNet50, self).__init__()

        self.in_channels = 64
        self.conv1       = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1         = nn.BatchNorm2d(64)
        self.relu        = nn.ReLU(inplace=True)
        self.maxpool     = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1      = self._make_layer(block, 64, layers[0])
        self.layer2      = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3      = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4      = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool     = nn.AdaptiveAvgPool2d((1, 1))
        self.fc          = self.fc = nn.Linear(512 * block.expansion, num_classes)


        # Initlaize weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsamples = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsamples = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsamples))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)
    

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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    
def resnet50(num_classes=1000):
    return ResNet50(BottleNeck, [3, 4, 6, 3], num_classes=num_classes)


model = resnet50(num_classes=1000)
print(model.forward(torch.randn(1, 3, 224, 224)).size())