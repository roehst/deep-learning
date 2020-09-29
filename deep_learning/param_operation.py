from abc import abstractmethod
from operation import Operation
from numpy import ndarray

class ParamOperation(Operation):
    def __init__(self, param: ndarray):
        self._param = param

    def backward(self, output_grad: ndarray) -> ndarray:
        self._input_grad = self.input_grad(output_grad)
        self._param_grad = self.param_grad(output_grad)
        return self._input_grad

    @abstractmethod
    def param_grad(self, output_grad: ndarray) -> ndarray:
        ...



