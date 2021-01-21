from abc import ABC, abstractmethod

class Compressor(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self):
        self._require_grad_idx = False

    @abstractmethod
    def compress(self, tensor):
        """Compresses a tensor with the given compression context, and then returns it with the context needed to decompress it."""

    @abstractmethod
    def decompress(self, tensors):
        """Decompress the tensor with the given decompression context."""

class EntropyCoder(ABC):
    """Interface for compressing and decompressing a given tensor."""

    def __init__(self):
        pass

    @abstractmethod
    def encode(self, seq):
        """Compresses a tensor with the given compression context, and then returns it with the context needed to decompress it."""

    @abstractmethod
    def decode(self, coded_set):
        """Decompress the tensor with the given decompression context."""

from fedlearning.stc import *
from fedlearning.entropy_coder import *

compressor_registry = {
    "stc": StcCompressor
}

entropy_coder_registry = {
    "golomb":   GolombCoder,
    "entropy":  IdealCoder
}