# # coding=utf-8
# # Copyright 2023 Beomi (L. Junbum)
# # Licensed under the Apache License, Version 2.0 (the "License")
""" PyTorch BitLinear Layer."""
import torch
import torch.nn as nn


class BitLinearNaive(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinearNaive, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5  # Small epsilon value to avoid division by zero and overflow
    
    def ternarize_weights(self,threshold=0.5):
        # 임계값 기반으로 가중치를 삼진화하여 -1, 0, 1 값으로 설정
        # x 값 중 threshold보다 작은 값은 0으로, 나머지는 부호에 따라 -1 또는 1로 설정
        ternarized_weights = torch.where((self.weight.abs() > threshold),
                                        torch.zeros_like(self.weight), 
                                        torch.sign(self.weight))
        return ternarized_weights

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
        # ternarize weights
        ternarized_weights = self.ternarize_weights()

        # Normal linear transformation with ternarized weights
        output = torch.nn.functional.linear(input, ternarized_weights, self.bias)

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

    def ste_ternarize(self, x, threshold=0.1):
        # 임계값을 기준으로 가중치를 -1, 0, 1로 삼진화합니다.
        ternarized_x = torch.where((x.abs() > threshold), torch.sign(x), torch.zeros_like(x))

        # STE를 사용하여 역전파 동안 삼진화를 우회합니다.
        # 순전파에서는 삼진화된 값을 사용하고, 역전파에서는 원래의 x 값을 기반으로 그라디언트를 계산합니다.
        ternarized_x = (ternarized_x - x).detach() + x
        return ternarized_x
    
    def ternarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        ternarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Ternarize each group using ste_ternarize (considering ste_ternarize method is updated for ternarization)
            ternarized_weights[start_idx:end_idx] = self.ste_ternarize(weight_group)

        return ternarized_weights

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
        # Ternarized weights (group-wise) using STE
        ternarized_weights = self.ternarize_weights_groupwise()

        # Normal linear transformation with ternarized weights
        output = torch.nn.functional.linear(input, ternarized_weights, self.bias)

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output


class BitLinearOptimized(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, num_groups=1):
        super(BitLinearOptimized, self).__init__(in_features, out_features, bias)
        self.num_groups = num_groups
        self.eps = 1e-5
        # 삼진법 가중치를 초기화하고 int8 형식으로 저장
        # 여기서 ternarize_weights는 적절한 삼진화 함수를 구현해야 함
        ternarized_weights = self.ternarize_weights(self.weight.data)
        self.register_buffer("quantized_weights", ternarized_weights.to(torch.int8))
        # Clear the original weights to save memory
        del self.weight

    @property
    def weight(self):
        # Return the dequantized weights when accessed
        return self.ternarize_weights()

    @weight.setter
    def weight(self, value):
        # 삼진화 로직을 사용하여 quantized_weights 업데이트
        ternarized_value = self.ternarize_weights(value)
        self.quantized_weights.data = ternarized_value.to(torch.int8)

    def ternarize_weights(self, weights,threshold = 0.5):
        # 예를 들어 임계값을 사용한 삼진화
        # 임계값은 경험적으로 설정하거나 최적화할 수 있음
        ternarized_weights = torch.where((weights.abs() > threshold), 
                                        torch.sign(weights),
                                        torch.zeros_like(weights))
        return ternarized_weights

    def dequantize_weights(self):
        # Convert quantized_weights back to bfloat16 and compute alpha for the weights
        bfloat16_weights = self.quantized_weights.to(torch.bfloat16)
        alpha = bfloat16_weights.mean()
        return bfloat16_weights * alpha

    
    def ste_ternarize(self, x, threshold=0.1):
        # 임계값을 기준으로 가중치를 -1, 0, 1로 삼진화합니다.
        ternarized_x = torch.where((x.abs() > threshold), torch.sign(x), torch.zeros_like(x))

        # STE를 사용하여 역전파 동안 삼진화를 우회합니다.
        # 순전파에서는 삼진화된 값을 사용하고, 역전파에서는 원래의 x 값을 기반으로 그라디언트를 계산합니다.
        ternarized_x = (ternarized_x - x).detach() + x
        return ternarized_x
    
    def ternarize_weights_groupwise(self):
        # Divide weights into groups
        group_size = self.weight.shape[0] // self.num_groups
        ternarized_weights = torch.zeros_like(self.weight)

        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size
            weight_group = self.weight[start_idx:end_idx]

            # Ternarize each group using ste_ternarize (considering ste_ternarize method is updated for ternarization)
            ternarized_weights[start_idx:end_idx] = self.ste_ternarize(weight_group)

        return ternarized_weights

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
        # Ternarized weights (group-wise) using STE
        ternarized_weights = self.ternarize_weights_groupwise()

        # Normal linear transformation with ternarized weights
        output = torch.nn.functional.linear(input.to(torch.bfloat16), ternarized_weights, self.bias)
        

        # Quantize activations group-wise
        output = self.quantize_activations_groupwise(output)

        return output
