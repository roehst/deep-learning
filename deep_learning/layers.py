from abc import ABC, abstractmethod

import numpy as np
from numpy import ndarray

from deep_learning.operations import (
    Operation,
    ParamOperation,
    Sigmoid,
    WeightMultiply,
    BiasAdd,
)


class Layer(ABC):
    def __init__(self, neurons: int):
        self._neurons = neurons
        self._first_pass = True
        self._params: List[ndarray] = []
        self._param_grads: List[ndarray] = []
        self._operations: List[Operation] = []

    @abstractmethod
    def setup_layer(self, input: ndarray):
        ...

    def forward(self, input: ndarray) -> ndarray:
        if self._first_pass:
            self.setup_layer(input)
            self._first_pass = False
        self._input = input
        self._output = self._input
        for op in self._operations:
            self._output = op.forward(self._output)
        return self._output

    def backward(self, output_grad: ndarray) -> ndarray:
        self._input_grad = output_grad
        for op in reversed(self._operations):
            self._input_grad = op.backward(self._input_grad)
        self.param_grads()
        return self._input_grad

    def param_grads(self):
        self._param_grads = []
        for op in self._operations:
            if issubclass(op.__class__, ParamOperation):
                self._param_grads.append(op._param_grad)

    def params(self):
        self._params = []
        for op in self._operations:
            if issubclass(op.__class__, ParamOperation):
                self._params.append(op._param)


class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation = Sigmoid()) -> None:
        super().__init__(neurons)
        self._activation = activation

    def setup_layer(self, input: ndarray):
        self._params = []
        self._params.append(np.random.randn(input.shape[1], self._neurons))
        self._params.append(np.random.randn(1, self._neurons))
        self._operations = [
            WeightMultiply(self._params[0]),
            BiasAdd(self._params[1]),
            self._activation,
        ]
