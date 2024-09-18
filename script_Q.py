import random
import gymnasium as gym
import numpy as np
import json

from src.Algorithms.SARSA import SARSA
from src.Algorithms.Q_learning import Q_learning

from src.Classes.Policy import Policy
from src.Classes.Agent import Agent
from src.Functions.run import run_static


desc = ["SFFF", "FHFH", "FFFH", "HFFG"]  # Same as the map called "4*4"
environment = gym.make(
    "FrozenLake-v1", desc=desc, is_slippery=True, render_mode="rgb_array"
)


Q_learning(environment, 0.6, 0.5, 0.98, 2000)
