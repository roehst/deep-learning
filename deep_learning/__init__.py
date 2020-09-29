from typing import List, Tuple

import numpy as np
import numpy.random as random
from numpy import ndarray
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.preprocessing import normalize

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

boston = load_boston()

X = boston["data"]
y = boston["target"]
D = np.column_stack([X, y])
df = pd.DataFrame(D, columns=list(boston["feature_names"]) + ["MDEV"])

X_train, X_test, y_train, y_test = train_test_split(X, y)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

X_train = normalize(X_train)
y_train = normalize(y_train)

X_test = normalize(X_test)
y_test = normalize(y_test)


trainer = Trainer(lr, SGD(learning_rate=0.0002))

trainer.fit(X_train, y_train, X_test, y_test, epochs=200)

for p in lr.params():
    print(p)
