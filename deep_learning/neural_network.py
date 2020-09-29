from numpy import ndarray
from typing import List

from layer import Layer
from loss import Loss

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

