import copy

import torch

class GradBuffer(object):
    def __init__(self, grad_dict, mode="copy"):
        self._grad_dict = copy.deepcopy(grad_dict)
        if mode == "zeros":
            for grad_name, grad_value in self._grad_dict.items():
                self._grad_dict[grad_name] = torch.zeros_like(grad_value)
        
    def __add__(self, grad_buffer):
        grad_dict = copy.deepcopy(self._grad_dict)
        for grad_name, grad_value in grad_dict.items():
            grad_dict[grad_name] = self._grad_dict[grad_name] + grad_buffer._grad_dict[grad_name]

        return GradBuffer(grad_dict)

    def __sub__(self, grad_buffer):
        grad_dict = copy.deepcopy(self._grad_dict)
        for grad_name, grad_value in grad_dict.items():
            grad_dict[grad_name] = self._grad_dict[grad_name] - grad_buffer._grad_dict[grad_name]

        return GradBuffer(grad_dict)

    def __mul__(self, rhs):
        grad_dict = copy.deepcopy(self._grad_dict)
        for grad_name, grad_value in grad_dict.items():
            grad_dict[grad_name] = rhs*self._grad_dict[grad_name]

        return GradBuffer(grad_dict)

    def push(self, grad_dict):
        self._grad_dict = copy.deepcopy(grad_dict)

    def grad_dict(self):
        return self._grad_dict

class WeightBuffer(object):
    def __init__(self, weight_dict, mode="copy"):
        self._weight_dict = copy.deepcopy(weight_dict)
        if mode == "zeros":
            for w_name, w_value in self._weight_dict.items():
                self._weight_dict[w_name].data = torch.zeros_like(w_value)
        
    def __add__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data + weight_buffer._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def __sub__(self, weight_buffer):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = self._weight_dict[w_name].data - weight_buffer._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def __mul__(self,rhs):
        weight_dict = copy.deepcopy(self._weight_dict)
        for w_name, w_value in weight_dict.items():
            weight_dict[w_name].data = rhs*self._weight_dict[w_name].data

        return WeightBuffer(weight_dict)

    def push(self, weight_dict):
        self._weight_dict = copy.deepcopy(weight_dict)

    def state_dict(self):
        return self._weight_dict

