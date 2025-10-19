import time
import torch
import torch.utils
from torch import nn
from brevitas import nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat

# However, quantization can potentially lead to a loss in model accuracy if not done carefully.
# Techniques like quantization-aware training (QAT) can help mitigate this accuracy loss
# by simulating the effects of quantization during the training process, allowing the
# model to adapt to the reduced precision.

# Define the number of bits for quantization
# Increased from 3 to 4 bits for better accuracy while still considering FHE constraints
N_BITS = 3
N_BITS_LATER = 4

class CNN(nn.Module):
    """Quantized CNN with flexible input size (channels + HxW)."""

    def __init__(self, n_classes, in_channels=1, image_size=None):
        super().__init__()

        # Quantization args
        qconv_args = {
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True
        }

        qlinear_args = {
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True
        }

        qidentity_args = {
            "bit_width": N_BITS,
            "act_quant": Int8ActPerTensorFloat
        }

        # Layers
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)

        self.conv1 = qnn.QuantConv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = qnn.QuantReLU(bit_width=N_BITS)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        self.quant_before_fc = qnn.QuantIdentity(**qidentity_args)

        # FC is dynamic → will be built after seeing data
        self.fc1 = None
        self.n_classes = n_classes
        self.qlinear_args = qlinear_args
        self.image_size = image_size  # store for logging/debugging

    def _build_fc(self, x):
        """Dynamically build the fully connected layer based on input size."""
        in_features = x.numel() // x.size(0)  # flatten size (after convs/pool)
        self.fc1 = qnn.QuantLinear(in_features, self.n_classes, **self.qlinear_args).to(x.device)

    def forward(self, x):
        x = self.quant_inp(x)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        x = self.quant_before_fc(x)
        x = x.flatten(1)

        if self.fc1 is None:
            self._build_fc(x)  # lazily build once

        x = self.fc1(x)
        return x

class CNN3(nn.Module):
    """Quantized CNN with 3 convolutional layers, FHE-ready."""

    def __init__(self, n_classes, in_channels=1, image_size=(28,28)):
        super().__init__()

        # Quantization args
        qconv_args = {
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True
        }

        qlinear_args = {
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True
        }

        qidentity_args = {
            "bit_width": N_BITS,
            "act_quant": Int8ActPerTensorFloat
        }

        # === Input quantization ===
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)

        # === Conv Block 1 ===
        self.conv1 = qnn.QuantConv2d(in_channels, 8, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = qnn.QuantReLU(bit_width=N_BITS)
        self.pool1 = nn.AvgPool2d(2)
        self.dropout1 = nn.Dropout(p=0.2)
        self.quant_pool1 = qnn.QuantIdentity(**qidentity_args)  # ensures conv2 sees quantized input

        # === Conv Block 2 ===
        self.conv2 = qnn.QuantConv2d(8, 16, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = qnn.QuantReLU(bit_width=N_BITS)
        self.pool2 = nn.AvgPool2d(2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.quant_pool2 = qnn.QuantIdentity(**qidentity_args)  # ensures conv3 sees quantized input

        # === Conv Block 3 ===
        self.conv3 = qnn.QuantConv2d(16, 32, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu3 = qnn.QuantReLU(bit_width=N_BITS)
        self.pool3 = nn.AvgPool2d(2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.quant_pool3 = qnn.QuantIdentity(**qidentity_args)  # ensures FC sees quantized input

        # === Fully connected ===
        self.quant_before_fc = qnn.QuantIdentity(**qidentity_args)

        # Compute input features for FC from image size
        h, w = image_size
        h = h // (2*2*2)  # 3 pooling layers
        w = w // (2*2*2)
        in_features = h * w * 32

        self.fc1 = qnn.QuantLinear(in_features, n_classes, **qlinear_args)

    def forward(self, x):
        x = self.quant_inp(x)

        # --- Conv Block 1 ---
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        x = self.quant_pool1(x)

        # --- Conv Block 2 ---
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        x = self.quant_pool2(x)

        # --- Conv Block 3 ---
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.dropout3(x)
        x = self.quant_pool3(x)

        # --- Fully connected ---
        x = self.quant_before_fc(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x
