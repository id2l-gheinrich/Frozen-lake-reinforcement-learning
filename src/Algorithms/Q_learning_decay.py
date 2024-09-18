import gymnasium as gym
import numpy as np
from nptyping import Float, NDArray, Shape

from src.Classes.Policy import Policy
from src.Classes.Agent import Agent
from src.Functions.epsilon_greedy_policy_factory import make_epsilon_greedy_policy
from src.Functions.reshape import reshape_one


def Q_learning(
    environment, epsilon_decay=0.01, alpha=0.1, gamma=0.99, epoch_number=5000
):
    # Get the observation space & the action space
    environment_space_length: int = environment.observation_space.n  # type: ignore
    action_space_length: int = environment.action_space.n  # type: ignore
    Q_sa = np.zeros((environment_space_length, action_space_length))

    def update_Qlearning(
        agent: Agent, state_index, action_index, next_state, reward: float = 0
    ):
        # Q[s, a] := Q[s, a] + α[r + γ . argmax_a {Q(s', a')} - Q(s, a)]
        best_next_action = np.argmax(Q_sa[next_state,])
        Q_sa[state_index, action_index] = Q_sa[state_index, action_index] + alpha * (
            reward
            + gamma * Q_sa[next_state, best_next_action]
            - Q_sa[state_index, action_index]
        )

    epsilon = 1
    for epoch in range(epoch_number):
        epsilon_greedy_policy = Policy(
            make_epsilon_greedy_policy(epsilon=epsilon, Q_sa=Q_sa),
            environment_space_length,
            action_space_length,
        )
        learning_agent = Agent(epsilon_greedy_policy, update_Qlearning)
        core_loop(environment, learning_agent)
        epsilon = epsilon * (1 - epsilon_decay)

    return Q_sa


def core_loop(environment, learning_agent):
    learning_agent.current_state_index = environment.reset()[0]  # Initial state
    stop = False
    action = learning_agent.pick_next()
    while not stop:
        """
        While Agent is not stopped
        Agent perform the action
        Agent pick an action from its state and its policy
        Agent update its trajectory: action taken, new state, new reward
        Update the stop condition if Goal reached or terminated
        """
        observation, reward, terminated, truncated, info = environment.step(action)
        next_action_index = learning_agent.pick_next(observation)
        learning_agent.trajectory.append(observation, next_action_index, float(reward))
        learning_agent.update(
            learning_agent.current_state_index, action, observation, reward
        )
        learning_agent.current_state_index = observation
        action = next_action_index
        stop = terminated or truncated


def has_converged(Q_sa, previous_shape: NDArray, difference_sequence: list):
    reshaped_Q_sa = reshape_one(Q_sa)
    difference_number = 0
    for row_index in range(previous_shape.shape[0]):
        is_same = (reshaped_Q_sa[row_index,] == previous_shape[row_index,]).all()
        if not is_same:
            difference_number += 1
        # print(reshaped_Q_sa[row_index, ])
        # print(previous_shape[row_index, ])
        # print(is_same)
    difference_sequence.append(difference_number)
    # We said we converge if the matrix shape didn't change over the last 100 iterations
    return len(difference_sequence) > 100 and sum(difference_sequence[-100:]) == 0
