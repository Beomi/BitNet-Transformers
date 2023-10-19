import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(BitLinear, self).__init__()
        
        # Initialize weight and bias
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights and bias
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialization based on linear layer's kaiming_uniform_
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    @staticmethod
    def binary_quantize(tensor):
        """Binarize the tensor to +1 or -1 using the signum function."""
        return torch.sign(tensor)
    
    @staticmethod
    def layer_normalization(tensor, epsilon=1e-5):
        """Compute Layer Normalization."""
        mean = tensor.mean(dim=1, keepdim=True)
        std = tensor.std(dim=1, keepdim=True)
        return (tensor - mean) / (std + epsilon)
    
    def forward(self, x):
        # Binarize the weights
        binarized_weight = self.binary_quantize(self.weight)
        
        # Compute Layer Normalization on input
        normalized_x = self.layer_normalization(x)
        
        # Compute output using binarized weights
        output = F.linear(normalized_x, binarized_weight, self.bias)
        
        return output

if __name__=="__main__":
    # Test the BitLinear layer
    bitlinear_layer = BitLinear(128, 64)
    input_tensor = torch.randn(32, 128)  # Batch of 32 samples with 128 features each
    output_tensor = bitlinear_layer(input_tensor)
    print(output_tensor.shape)  # Expected: [32, 64]

