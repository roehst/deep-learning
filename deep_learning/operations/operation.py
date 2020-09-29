from abc import ABC, abstractmethod

from numpy import ndarray


class Operation(ABC):
    def forward(self, input: ndarray) -> ndarray:
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
