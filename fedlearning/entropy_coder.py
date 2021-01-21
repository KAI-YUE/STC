import numpy as np

# PyTorch libraries
import torch

from fedlearning import EntropyCoder
import fedlearning.constant as const

class IdealCoder (EntropyCoder):
    def __init__(self, config):
        self.total_codewords = 3    # ternary quantization

    def encode(self, seq):
        """
        Simulate an ideal entropy coding to a quantized tensor 
        without actually coding the tensor array. 
        """
        histogram = torch.histc(seq, bins=self.total_codewords, min=-1, max=1) 
        total_symbols = torch.sum(histogram).item()
        entropy = self._entropy((histogram.to(torch.float)/total_symbols).detach().cpu().numpy())
        
        original_bits = total_symbols * const.FLOAT_BIT
        compressed_bits = self._compressed_bits(total_symbols, entropy)

        coded_set = dict(code=seq,
                    compressed_bits=compressed_bits,
                    original_bits=original_bits)

        return coded_set
                    
    def decode(self, coded_set):
        """
        Simulate an ideal entropy decoding without actually decoding the tensor array. 
        """
        return coded_set["code"]

    def _compressed_bits(self, total_symbols, ratio):
        magnitude_bits = 1 * const.FLOAT_BIT
        
        return total_symbols*ratio+ magnitude_bits

    @staticmethod
    def _entropy(histogram):
        entropy = 0

        for i, prob in enumerate(histogram):
            if prob == 0:
                continue
            entropy += -prob * np.log2(prob)

        return entropy

# class GolombCoder (EntropyCoder):
#     def __init__(self, config):
#         self.sparsity = config.sparsity
