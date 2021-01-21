import numpy as np
import torch

# My libraries
from fedlearning import Compressor

class StcCompressor(Compressor):

    def __init__(self, config):
        super().__init__()
        self.sparsity = config.sparsity

    def compress(self, tensor, **kwargs):
        """
        Compress the input tensor with signSGD and simulate the saved data volume in bit.

        Args,
            tensor (torch.tensor): the input tensor.
        """
        k = np.ceil(tensor.numel()*self.sparsity).astype(int)
        top_k_element, top_k_index = torch.kthvalue(-tensor.abs().flatten(), k)
        tensor_masked = (tensor.abs() > -top_k_element) * tensor

        magnitude = (1/k) * tensor_masked.abs().sum()
        coded_set = dict(magnitude=magnitude, quantized_arr=tensor_masked.sign())

        return coded_set 

    def decompress(self, coded_set):
        """Decode the signs to float format """
        quantized_arr = coded_set["quantized_arr"]
        magnitude = coded_set["magnitude"]
        return magnitude*quantized_arr

    def normalize_aggregation(self, tensor, threshold):
        normalized_tensor = (tensor >= threshold).to(torch.float) * 2 - 1
        return normalized_tensor
    
