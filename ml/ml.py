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


