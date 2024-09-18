import gymnasium as gym
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import random

learning_rate = 0.05
gamma = 0.95
# epsilon = 0.6
epsilon_decay = 0.01
epsilon_factor = 1 - epsilon_decay
epoch_number = 100

desc = ["SFFF", "FHFH", "FFFH", "HFFG"]  # Same as the map called "4*4"
environment = gym.make(
    "FrozenLake-v1", desc=desc, is_slippery=False, render_mode="rgb_array"
)

environment.reset(seed=0)

environment_space_dimension: int = environment.observation_space.n  # type: ignore
action_space_length: int = environment.action_space.n  # type: ignore

optimizer = keras.optimizers.SGD(learning_rate=learning_rate)
model = keras.Sequential(
    [
        layers.Input(shape=(environment_space_dimension,)),
        layers.Dense(24, activation="relu"),
        layers.Dense(24, activation="relu"),
        layers.Dense(action_space_length, activation="linear"),
    ]
)

model.compile(loss="mse", optimizer=optimizer)  # type: ignore


def update(state_index, action_index, next_state, reward: float = 0, done=False):
    state_vector = np.zeros(environment_space_dimension)
    state_vector[state_index] = 1
    state_as_matrix = np.array(
        [state_vector],
    )

    # print(next_state)
    next_state_vector = np.zeros(environment_space_dimension)
    next_state_vector[int(next_state)] = 1
    next_state_as_matrix = np.array(
        [next_state_vector],
    )
    # print(state_as_matrix)
    target = model.predict(state_as_matrix, verbose=0)[0]  # type: ignore
    if done:
        target[action_index] = reward
    else:
        target[action_index] = reward + gamma * np.max(
            model.predict(next_state_as_matrix, verbose=0)  # type: ignore
        )
    inputs = state_as_matrix
    outputs = np.array([target])
    return model.fit(inputs, outputs, epochs=1, verbose=0, batch_size=1)  # type: ignore


def get_best_action(state, epsilon: None | float = None):
    # print("in get_best_action")
    if epsilon and np.random.rand() <= epsilon:
        # The agent acts randomly
        action = random.randrange(action_space_length)
        # print("acted randomly", action)
        return action

    state_vector = np.zeros(environment_space_dimension)
    state_vector[state] = 1
    state_as_matrix = np.array(
        [state_vector],
    )
    # Predict the reward value based on the given state
    act_values = model.predict(state_as_matrix, verbose=0)  # type: ignore

    # Pick the action based on the predicted reward
    action = np.argmax(act_values[0])
    # print("Picked action ", action)
    return action


scores = []
epsilon = 1
for epoch in range(epoch_number):
    state = environment.reset()[0]  # Initial state
    stop = False
    score: float = 0
    steps: float = 0
    epsilon *= epsilon_factor
    while not stop:
        steps += 1
        action = get_best_action(state, epsilon)
        observation, reward, terminated, truncated, info = environment.step(action)
        # print("reward: ", reward)
        score += float(reward)
        stop = terminated or truncated
        update(state, action, observation, float(reward), stop)
        state = observation
        if stop:
            print(
                "episode: {}/{}, moves: {}, score: {}, truncated: {}, terminated: {}".format(
                    epoch + 1, epoch_number, steps, score, truncated, terminated
                )
            )
            scores.append(score)
            break
        if epoch % 100 == 0:  # print log every 100 episode
            print(
                "episode: {}/{}, moves: {}, score: {}".format(
                    epoch, epoch_number, steps, score
                )
            )
print(scores)
