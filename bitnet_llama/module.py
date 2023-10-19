# # coding=utf-8
# # Copyright 2023 Beomi (L. Junbum)
# # Licensed under the Apache License, Version 2.0 (the "License")
# """ PyTorch BitLinear Layer."""
# import math

# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class BitLinear(nn.Module):
#     def __init__(self, in_features, out_features, bias=True):
#         super(BitLinear, self).__init__()

#         # Initialize weight and bias
#         self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
#         if bias:
#             self.bias = nn.Parameter(torch.Tensor(out_features))
#         else:
#             self.register_parameter("bias", None)

#         # Initialize weights and bias
#         self.reset_parameters()

#     def reset_parameters(self):
#         # Initialization based on linear layer's kaiming_uniform_
#         nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
#         if self.bias is not None:
#             fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
#             bound = 1 / math.sqrt(fan_in)
#             nn.init.uniform_(self.bias, -bound, bound)

#     @staticmethod
#     def binary_quantize(tensor):
#         """Binarize the tensor to +1 or -1 using the signum function."""
#         return torch.sign(tensor)

#     @staticmethod
#     def layer_normalization(tensor, epsilon=1e-5):
#         """Compute Layer Normalization."""
#         mean = tensor.mean(dim=1, keepdim=True)
#         std = tensor.std(dim=1, keepdim=True)
#         return (tensor - mean) / (std + epsilon)

#     def forward(self, x):
#         # Binarize the weights
#         binarized_weight = self.binary_quantize(self.weight)

#         # Compute scaling factor
#         gamma = (1 / (self.weight.size(0) * self.weight.size(1))) * self.weight.norm(
#             p=1
#         )

#         # Compute Layer Normalization on input
#         normalized_x = self.layer_normalization(x)

#         # Compute output using binarized weights and scale by gamma
#         output = gamma * F.linear(normalized_x, binarized_weight, self.bias)

#         return output


# if __name__ == "__main__":
#     # Test the updated BitLinear layer again
#     bitlinear_layer = BitLinear(128, 64)
#     input_tensor = torch.randn(32, 128)  # Batch of 32 samples with 128 features each
#     output_tensor = bitlinear_layer(input_tensor)
#     output_tensor.shape  # Expected: [32, 64]

import torch
import torch.nn as nn


class BitLinearNaive(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinearNaive, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5  # Small epsilon value to avoid division by zero and overflow

    def binarize_weights(self):
        alpha = self.weight.mean()
        binarized_weights = torch.sign(self.weight - alpha)
        return binarized_weights

    def quantize_activations(self, x, b=8):
        Q_b = 2 ** (b - 1)
        gamma = x.abs().max()
        quantized_x = torch.clamp(
            x * Q_b / (gamma + self.eps), -Q_b + self.eps, Q_b - self.eps
        )
        return quantized_x

    def scale_activations(self, x, b=8):
        Q_b = 2 ** (b - 1)
        eta = x.min()
        gamma = x.abs().max()
        scaled_x = torch.clamp(
            (x - eta) * Q_b / (gamma + self.eps), self.eps, Q_b - self.eps
        )
        return scaled_x

    def forward(self, input):
        # Binarize weights
        binarized_weights = self.binarize_weights()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations (before non-linear functions like ReLU)
        output = self.quantize_activations(output)

        # For the sake of demonstration, we'll also include the scaling step.
        # In practice, this would be done before a non-linear function in a forward pass.
        output = self.scale_activations(output)

        return output


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinear, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5

    def ste_binarize(self, x):
        # Apply the sign function for binarization
        binarized_x = torch.sign(x)
        # Use STE: during backward pass, we bypass the binarization
        binarized_x = (binarized_x - x).detach() + x
        return binarized_x

    def binarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        binarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Binarize each group using STE
            alpha_g = weight_group.mean()
            binarized_weights[start_idx:end_idx] = self.ste_binarize(
                weight_group - alpha_g
            )

        return binarized_weights

    def quantize_activations_groupwise(self, x, b=8):
        Q_b = 2 ** (b - 1)

        # Divide activations into groups
        group_size = x.shape[0] // self.num_groups
        quantized_x = torch.zeros_like(x)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            activation_group = x[start_idx:end_idx]

            # Quantize each group
            gamma_g = activation_group.abs().max()
            quantized_x[start_idx:end_idx] = torch.clamp(
                activation_group * Q_b / (gamma_g + self.eps),
                -Q_b + self.eps,
                Q_b - self.eps,
            )

        return quantized_x

    def forward(self, input):
        # Binarize weights (group-wise) using STE
        binarized_weights = self.binarize_weights_groupwise()

        # Normal linear transformation with binarized weights
        output = torch.nn.functional.linear(input, binarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


# # Test the BitLinear layer with group quantization and STE
# input_tensor = torch.randn(10, 20)
# bitlinear_layer_ste = BitLinearWithGroupQuantizationAndSTE(20, 30, num_groups=5)
# output_tensor_ste = bitlinear_layer_ste(input_tensor)
# output_tensor_ste.shape


# # Test the BitLinear layer
# input_tensor = torch.randn(10, 20)
# bitlinear_layer = BitLinear(20, 30)
# output_tensor = bitlinear_layer(input_tensor)
# output_tensor.shape
