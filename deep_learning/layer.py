from abc import ABC, abstractmethod

from numpy import ndarray

from operation import Operation
from param_operation import ParamOperation


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
