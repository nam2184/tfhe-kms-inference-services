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

class CNN8(nn.Module):
    """A small quantized CNN to classify the sklearn digits dataset, optimized for FHE."""

    def __init__(self, n_classes):
        super().__init__()

        # Quantization arguments for Conv and Linear layers
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

        # Quantized input layer
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)

        # Quantized convolution layer with activation
        self.conv1 = qnn.QuantConv2d(1, 8, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn1 = nn.BatchNorm2d(8)
        self.relu1 = qnn.QuantReLU(bit_width=N_BITS)

        # Pooling and dropout
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout1 = nn.Dropout(p=0.5)

        # Additional quantization before flattening and FC layer
        self.quant_before_fc = qnn.QuantIdentity(**qidentity_args)

        # Fully connected quantized layer
        self.fc1 = qnn.QuantLinear(128, n_classes, **qlinear_args)

    def forward(self, x):
        # Quantize the input
        x = self.quant_inp(x)

        # Convolution -> BatchNorm -> Activation -> Pooling -> Dropout
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool(x)
        x = self.dropout1(x)

        # Quantize again before flattening and fully connected layer
        x = self.quant_before_fc(x)

        # Flatten and feed into final linear layer
        x = x.flatten(1)  # Flatten all but batch dimension
        x = self.fc1(x)

        return x


class CNN28(nn.Module):
    """A small quantized CNN to classify 28x28 images, optimized for FHE."""

    def __init__(self, n_classes):
        super().__init__()
        
        # Quantization arguments for Conv and Linear layers
        qconv_args = {
            "weight_bit_width": N_BITS,  # Set the bit width for weights to N_BITS (4 in this case)
            "weight_quant": Int8WeightPerTensorFloat,  # Use 8-bit integer quantization for weights
            "bias": True,  # Include bias in the layer
            "bias_quant": None,  # No quantization for bias (full precision)
            "narrow_range": True  # Use a narrower range for quantization, improving precision
        }
        
        # Similar to qconv_args, but for fully connected (Linear) layers
        qlinear_args = {
            "weight_bit_width": N_BITS,
            "weight_quant": Int8WeightPerTensorFloat,
            "bias": True,
            "bias_quant": None,
            "narrow_range": True
        }

        # Arguments for quantizing the input (identity layer)
        qidentity_args = {
            "bit_width": N_BITS,  # Set the bit width for activations to N_BITS
            "act_quant": Int8ActPerTensorFloat  # Use 8-bit integer quantization for activations
        }

        # Quantized input layer to ensure the input is compatible with quantized operations
        self.quant_inp = qnn.QuantIdentity(**qidentity_args)

        # First Convolutional Block (28x28 -> 14x14 after pooling)
        self.conv1 = qnn.QuantConv2d(1, 16, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = qnn.QuantReLU(bit_width=N_BITS)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # Reduces 28x28 to 14x14
        self.dropout1 = nn.Dropout(p=0.5)

        # Second Convolutional Block (14x14 -> 7x7 after pooling)
        self.conv2 = qnn.QuantConv2d(16, 32, kernel_size=3, stride=1, padding=1, **qconv_args)
        self.bn2 = nn.BatchNorm2d(32)
        self.relu2 = qnn.QuantReLU(bit_width=N_BITS)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)  # Reduces 14x14 to 7x7
        self.dropout2 = nn.Dropout(p=0.5)

        # After pooling, the feature map size is 7x7x32 = 1568
        self.fc1 = qnn.QuantLinear(7 * 7 * 32, n_classes, **qlinear_args)

    def forward(self, x):
        x = self.quant_inp(x)
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc1(x)
        return x
