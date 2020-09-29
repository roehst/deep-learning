from abc import ABC, abstractmethod

from numpy import ndarray
import numpy as np


class Operation(ABC):
    def forward(self, input: ndarray) -> ndarray:
        assert not np.any(np.isnan(input))
        self._input = input
        self._output = self.output()
        return self._output

    def backward(self, output_grad: ndarray) -> ndarray:
        self._input_grad = self.input_grad(output_grad)
        return self._input_grad

    @abstractmethod
    def output(self) -> ndarray:
        ...

    @abstractmethod
    def input_grad(self, output_grad: ndarray) -> ndarray:
        ...


class Linear(Operation):
    def output(self) -> ndarray:
        return self._input

    def input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad


class Sigmoid(Operation):
    def output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self._input))

    def input_grad(self, output_grad: ndarray) -> ndarray:
        sigmoid_backward = self._output * (1.0 - self._output)
        return sigmoid_backward * output_grad


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


class WeightMultiply(ParamOperation):
    def output(self) -> ndarray:
        return np.dot(self._input, self._param)

    def input_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(output_grad, np.transpose(self._param, (1, 0)))

    def param_grad(self, output_grad: ndarray) -> ndarray:
        return np.dot(np.transpose(self._input, (1, 0)), output_grad)


class BiasAdd(ParamOperation):
    def output(self) -> ndarray:
        return self._input + self._param

    def input_grad(self, output_grad: ndarray) -> ndarray:
        return output_grad * np.ones_like(self._input)

    def param_grad(self, output_grad: ndarray) -> ndarray:
        param_grad = np.ones_like(self._param) * output_grad
        return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])
