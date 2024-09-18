from __future__ import annotations  # See https://stackoverflow.com/a/42845998

from typing import Callable
from nptyping import NDArray, Shape, Int, Float
import random
import numpy as np
import keras


class Policy(object):

    def __init__(
        self,
        next_function: Callable[[Policy, int], int],
        state_dimension: int,
        action_dimension: int,
    ):
        self._next_function = next_function
        self.state_dimension = state_dimension
        self.action_dimension = action_dimension

    def pick_next(self, state_index: int) -> int:
        assert state_index >= 0 < self.state_dimension
        return self._next_function(self, state_index)

    #     assert state_index >= 0 < self.state_dimension
    #     rand = random.random()
    #     action_subset = self.policy_array[state_index, ]
    #     # See https://stackoverflow.com/questions/16243955/numpy-first-occurrence-of-value-greater-than-existing-value
    #     return(int(np.searchsorted(np.cumsum(action_subset), rand)))

    @staticmethod
    def buildOptimalPolicyFromQMatrix(
        Q_sa: NDArray[Shape["StateDimension, ActionDimension"], Float]
    ):
        state_dimension = Q_sa.shape[0]
        action_dimension = Q_sa.shape[1]
        deterministic_policy = Policy(
            lambda policy, state: int(np.argmax(Q_sa[state])),
            state_dimension,
            action_dimension,
        )
        return deterministic_policy

    @staticmethod
    def buildOptimalPolicyFromNeuralNetwork(neural_network: keras.Sequential):
        state_dimension = neural_network.layers[0].input_shape
        action_dimension = neural_network.layers[-1].input_shape
        deterministic_policy = Policy(
            lambda policy, state: int(
                np.argmax(neural_network.predict(np.array([state])))
            ),
            state_dimension,
            action_dimension,
        )
        return deterministic_policy
