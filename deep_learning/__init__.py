from typing import List, Tuple

import numpy as np
import numpy.random as random
from numpy import ndarray
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

from operation import Operation
from param_operation import ParamOperation


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


class Loss(ABC):
    def forward(self, prediction: ndarray, target: ndarray) -> float:
        self._prediction = prediction
        self._target = target
        self._loss_value = self.output()
        return self._loss_value

    def backward(self) -> ndarray:
        self._input_grad = self.input_grad()
        return self._input_grad

    @abstractmethod
    def output(self) -> float:
        ...

    @abstractmethod
    def input_grad(self) -> ndarray:
        ...


class MeanSquaredError(Loss):
    def output(self) -> float:
        return (
            np.sum(np.power(self._prediction - self._target, 2.0))
            / self._prediction.shape[0]
        )

    def input_grad(self) -> ndarray:
        return 2.0 * (self._prediction - self._target) / self._prediction.shape[0]


class NeuralNetwork:
    def __init__(
        self,
        layers: List[Layer],
        loss: Loss,
    ) -> None:
        self._layers = layers
        self._loss = loss

    def forward(self, X: ndarray) -> ndarray:
        out = X
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self, loss_grad: ndarray):
        grad = loss_grad
        for layer in reversed(self._layers):
            grad = layer.backward(grad)

    def train_batch(self, X: ndarray, y: ndarray) -> float:
        predictions = self.forward(X)
        loss = self._loss.forward(predictions, y)
        self.backward(self._loss.backward())
        return loss

    def params(self):
        for layer in self._layers:
            yield from layer._params

    def param_grads(self):
        for layer in self._layers:
            yield from layer._param_grads


class Optimizer(ABC):
    def __init__(self, learning_rate: float = 0.01) -> None:
        self._learning_rate = learning_rate

    @abstractmethod
    def step(self) -> None:
        ...


class SGD(Optimizer):
    def step(self) -> None:
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self._learning_rate * param_grad


def permute_data(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
    return (X, y)


class Trainer:
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        self._net = net
        self._optim = optim
        setattr(self._optim, "net", self._net)

    def generate_batches(
        self, X: ndarray, y: ndarray, size: int = 32
    ) -> Tuple[ndarray]:

        N = X.shape[0]

        for ii in range(0, N, size):
            X_batch, y_batch = X[ii : ii + size], y[ii : ii + size]

            yield X_batch, y_batch

    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        restart: bool = True,
    ):
        if restart:
            for layer in self._net._layers:
                layer._first_pass = True

        for e in range(epochs):
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self._net.train_batch(X_batch, y_batch)
                self._optim.step()
            if (e + 1) % eval_every == 0:
                test_preds = self._net.forward(X_test)
                loss = self._net._loss.forward(test_preds, y_test)
                print(f"Validation loss after {e+1} epochs is {loss:.8f}")


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
