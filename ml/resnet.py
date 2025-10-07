import torch
import torch.nn as nn
from brevitas import nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

N_BITS = 3  # Low-bit quantization for FHE-friendly deployment

def conv3x3(in_planes, out_planes, stride=1):
    return qnn.QuantConv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1,
        weight_bit_width=N_BITS, weight_quant=Int8WeightPerTensorFloat, bias=True, narrow_range=True
    )

def conv1x1(in_planes, out_planes, stride=1):
    return qnn.QuantConv2d(
        in_planes, out_planes, kernel_size=1, stride=stride,
        weight_bit_width=N_BITS, weight_quant=Int8WeightPerTensorFloat, bias=True, narrow_range=True
    )

class LiteBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(bit_width=N_BITS)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        # quantizer for the block output (before Add / residual)
        self.quant_after_bn2 = qnn.QuantIdentity(bit_width=N_BITS, act_quant=Int8ActPerTensorFloat)

        # downsample path (if present) should include a quantizer after its BN as well:
        self.downsample = downsample
        self.relu2 = qnn.QuantReLU(bit_width=N_BITS)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # quantize the block's main path before the Add
        out = self.quant_after_bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)  # ensure downsample contains its own QuantIdentity (see _make_layer)

        out = out + identity
        out = self.relu2(out)
        return out


class LiteResNet(nn.Module):
    def __init__(self, n_classes=2, in_channels=3, input_size=48,
                 block=LiteBasicBlock, layers=[1, 1, 1]):
        super().__init__()
        self.inplanes = 16
        self.quant_inp = qnn.QuantIdentity(bit_width=N_BITS, act_quant=Int8ActPerTensorFloat)

        # Stem
        self.stem = nn.Sequential(
            qnn.QuantConv2d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1,
                            weight_bit_width=N_BITS, weight_quant=Int8WeightPerTensorFloat,
                            bias=True, narrow_range=True),
            nn.BatchNorm2d(self.inplanes),
            qnn.QuantReLU(bit_width=N_BITS),
            nn.MaxPool2d(2),
            qnn.QuantIdentity(bit_width=N_BITS, act_quant=Int8ActPerTensorFloat)
        )

        # Residual layers
        self.layer1 = self._make_layer(block, 16, layers[0])
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2)

        # Compute feature map size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size, input_size)
            out = self._forward_features(dummy)
            feature_dim = out.shape[1] * out.shape[2] * out.shape[3]

        self.quant_out = qnn.QuantIdentity(bit_width=N_BITS, act_quant=Int8ActPerTensorFloat)
        self.fc = qnn.QuantLinear(
            feature_dim, n_classes,
            weight_bit_width=N_BITS, weight_quant=Int8WeightPerTensorFloat,
            bias=True, narrow_range=True
        )

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
                qnn.QuantIdentity(bit_width=N_BITS, act_quant=Int8ActPerTensorFloat)
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_features(self, x):
        x = self.quant_inp(x)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = self.quant_out(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
