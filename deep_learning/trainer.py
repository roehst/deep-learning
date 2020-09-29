from typing import Tuple, Generator

from numpy import ndarray

from deep_learning.neural_network import NeuralNetwork
from deep_learning.optimizer import Optimizer


def permute_data(X: ndarray, y: ndarray) -> Tuple[ndarray, ndarray]:
    return (X, y)


class Trainer:
    def __init__(self, net: NeuralNetwork, optim: Optimizer) -> None:
        self._net = net
        self._optim = optim
        setattr(self._optim, "net", self._net)

    def generate_batches(
        self, X: ndarray, y: ndarray, size: int = 32
    ) -> Generator[Tuple[ndarray, ndarray], None, None]:

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
