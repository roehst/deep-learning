from typing import List, Tuple

import numpy as np
import numpy.random as random
from numpy import ndarray
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

from deep_learning.operations import Operation, ParamOperation
from deep_learning.loss import Loss
from deep_learning.layer import Layer
from deep_learning.neural_network import NeuralNetwork
from deep_learning.optimizer import Optimizer
from deep_learning.trainer import Trainer


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


class MeanSquaredError(Loss):
    def output(self) -> float:
        return (
            np.sum(np.power(self._prediction - self._target, 2.0))
            / self._prediction.shape[0]
        )

    def input_grad(self) -> ndarray:
        return 2.0 * (self._prediction - self._target) / self._prediction.shape[0]


class SGD(Optimizer):
    def step(self) -> None:
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self._learning_rate * param_grad


lr = NeuralNetwork(
    layers=[Dense(neurons=1, activation=Linear())], loss=MeanSquaredError()
)

n = 200
X = np.random.normal(0, 1, (n, 1))
alpha = 2.0
beta = np.array([[1.25]])
error = np.random.normal(0, 1, (n, 1))
y = alpha + beta * X + error

index = list(range(50))

X_train, X_test, y_train, y_test = train_test_split(X, y)

trainer = Trainer(lr, SGD(learning_rate=0.002))

trainer.fit(X_train, y_train, X_test, y_test, epochs=5000)

for p in lr.params():
    print(p)
