from typing import List, Tuple

import numpy as np
import numpy.random as random
from numpy import ndarray
from sklearn.model_selection import train_test_split

from abc import ABC, abstractmethod

from deep_learning.operations import (
    Operation,
    ParamOperation,
    WeightMultiply,
    BiasAdd,
    Linear,
)
from deep_learning.loss import Loss, MeanSquaredError
from deep_learning.layers import Layer, Dense
from deep_learning.neural_network import NeuralNetwork
from deep_learning.optimizer import Optimizer, SGD
from deep_learning.trainer import Trainer


lr = NeuralNetwork(
    layers=[Dense(neurons=1, activation=Linear())], loss=MeanSquaredError()
)

n = 200
k = 2
X = np.random.normal(0, 1, (n, k))
alpha = 2.0
beta = np.random.normal(1, 1, (k, 1))
error = np.random.normal(0, 1, (n, 1))
y = alpha + np.dot(X, beta) + error

X_train, X_test, y_train, y_test = train_test_split(X, y)

trainer = Trainer(lr, SGD(learning_rate=0.002))

trainer.fit(X_train, y_train, X_test, y_test, epochs=5000)

for p in lr.params():
    print(p)
