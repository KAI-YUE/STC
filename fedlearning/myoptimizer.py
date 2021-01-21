import logging
import copy
from collections import OrderedDict

# PyTorch libraries
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
from torch.optim import Optimizer

# My libraries
from deeplearning import UserDataset
from config.utils import parse_dataset_type
from fedlearning import compressor_registry, entropy_coder_registry
from fedlearning.buffer import GradBuffer

class LocalUpdater(object):
    def __init__(self, user_resource, config, **kwargs):
        """Construct a local updater for a user.

        Args:
            user_resources(dict):   a dictionary containing images and labels listed as follows. 
                - images (ndarry):  training images of the user.
                - labels (ndarray): training labels of the user.

            config (class):         global configuration containing info listed as follows:
                - lr (float):       learning rate for the user.
                - batch_size (int): batch size for the user. 
                - device (str):     set 'cuda' or 'cpu' for the user. 
                - predictor (str):  predictor type.
                - quantizer (str):  quantizer type.
        """
        
        try:
            self.lr = user_resource["lr"]
            self.momentum = user_resource["momentum"]
            self.weight_decay = user_resource["weight_decay"]
            self.batch_size = user_resource["batch_size"]
            self.device = user_resource["device"]
            
            assert("images" in user_resource)
            assert("labels" in user_resource)
        except KeyError:
            logging.error("LocalUpdater Initialization Failure! Input should include `lr`, `batch_size`!") 
        except AssertionError:
            logging.error("LocalUpdater Initialization Failure! Input should include samples!") 

        dataset_type = parse_dataset_type(config)

        if config.imbalance:
            sampler = WeightedRandomSampler(user_resource["sampling_weight"], 
                                    num_samples=user_resource["sampling_weight"].shape[0])

            self.sample_loader = \
                DataLoader(UserDataset(user_resource["images"], 
                                user_resource["labels"],
                                dataset_type), 
                            sampler=sampler,
                            # sampler=None,
                            batch_size=self.batch_size)
        else:
            self.sample_loader = \
                DataLoader(UserDataset(user_resource["images"], 
                            user_resource["labels"],
                            dataset_type), 
                    sampler=None, 
                    batch_size=self.batch_size,
                    shuffle=True)

        self.criterion = nn.CrossEntropyLoss()
        self.compressor = compressor_registry[config.compressor](config)
        self.compressed_diff_set = None

        self.entropy_coder = entropy_coder_registry[config.entropy_coder](config)

    def local_step(self, model, local_residual):
        """Perform local update tau times.

        Args,
            model(nn.module):       the global model.
        """
        optimizer = optim.SGD(model.parameters(), lr=self.lr, momentum=self.momentum)

        for sample in self.sample_loader:
            image = sample["image"].to(self.device)
            label = sample["label"].to(self.device)
            optimizer.zero_grad()

            output = model(image)
            loss = self.criterion(output, label)
            loss.backward()

            compressed_diff_set = self.compress_grad(model, local_residual)
            optimizer.zero_grad()
            break
        
        self.compressed_diff_set = compressed_diff_set

    def compress_grad(self, model, local_residual):
        self.uncompressed_diff = OrderedDict()
        compressed_diff_set = OrderedDict()
        named_modules = model.named_modules()
        next(named_modules)
        
        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif module.weight is None:
                continue
            self.uncompressed_diff[module_name + ".weight"] = -module.weight.grad+local_residual[module_name+".weight"]
            compressed_diff_set[module_name + ".weight"] = self.compressor.compress(self.uncompressed_diff[module_name + ".weight"])
            
            if module.bias is None:
                continue
            self.uncompressed_diff[module_name + ".bias"] = -module.bias.grad+local_residual[module_name+".bias"]
            compressed_diff_set[module_name + ".bias"] = self.compressor.compress(self.uncompressed_diff[module_name + ".bias"])

        return compressed_diff_set

    def uplink_transmit(self, local_residual):
        """Simulate the transmission of residual between local updated weight and local received initial weight.
        """ 
        coded_grad_sets = OrderedDict()
        compressed_diff = OrderedDict()
        for w_name, quantized_set in self.compressed_diff_set.items():
            compressed_diff[w_name] = self.compressor.decompress(quantized_set)
            coded_grad_sets[w_name] = self.entropy_coder.encode(quantized_set["quantized_arr"])

            local_residual[w_name] = self.uncompressed_diff[w_name] - compressed_diff[w_name]

        return dict(compressed_diff=compressed_diff, 
                    coded_grad_sets=coded_grad_sets)


class GlobalUpdater(object):
    def __init__(self, config, initial_model, **kwargs):
        """Construct a global updater for a server.

        Args:
            config (class):              global configuration containing info listed as follows:

            initial_model (OrderedDict): initial model state_dict
        """
        self.num_users = config.users
        self.lr = config.lr

        self.accumulated_delta = None
        self.compressor = compressor_registry[config.compressor](config)

        self.original_bits = 0
        self.compressed_bits = 0

    def global_step(self, model, local_packages, **kwargs):
        """Perform a global update with collocted coded info from local users.
        """
        self.original_bits = 0
        self.compressed_bits = 0

        accumulated_grad = GradBuffer(local_packages[0]["compressed_diff"], mode="zeros") 
        for i, package in enumerate(local_packages):
            accumulated_grad += GradBuffer(package["compressed_diff"])
            for w_name, w_value in package["compressed_diff"].items():
                self.original_bits += package["coded_grad_sets"][w_name]["original_bits"]
                self.compressed_bits += package["coded_grad_sets"][w_name]["compressed_bits"]

        neg_grad_dict = accumulated_grad.grad_dict()

        named_modules = model.named_modules()
        next(named_modules)

        for module_name, module in named_modules:
            if not hasattr(module, "weight"):
                continue
            elif module.weight is None:
                continue
            module.weight.data += self.lr*neg_grad_dict[module_name + ".weight"]
            
            if module.bias is None:
                continue
            module.bias.data += self.lr*neg_grad_dict[module_name + ".bias"]

    @property
    def compress_ratio(self):
        try:
            return self.original_bits/self.compressed_bits
        except ZeroDivisionError:
            return None