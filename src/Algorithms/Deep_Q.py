import gymnasium as gym
import numpy as np
import tensorflow as tf
import keras
from keras import layers
import random

from src.Classes.Policy import Policy
from src.Classes.Agent import Agent

# TODO from tensorflow.keras.models import load_model
# # Sauvegarder le modèle original
# model.save('path_to_my_model.h5')

# # Charger une nouvelle instance du modèle
# new_model = load_model('path_to_my_model.h5')


# TODO séparer le train de l'utilisation du modèle
def Deep_Q(
    environment,
    epsilon=0.1,
    alpha=0.1,
    gamma=0.99,
    epoch_number=8000,
    pretrained_model: None | keras.Sequential = None,
    training_policy: bool = True,
):
    # Get the observation space & the action space
    environment_space_length: int = environment.observation_space.n  # type: ignore
    action_space_length: int = environment.action_space.n  # type: ignore

    if pretrained_model is None:
        optimizer = keras.optimizers.SGD(learning_rate=alpha)
        model = keras.Sequential(
            [
                layers.Input(shape=(1,)),
                layers.Dense(24, activation="relu"),
                layers.Dense(24, activation="relu"),
                layers.Dense(action_space_length, activation="linear"),
            ]
        )
        # model.add(layers.Input(shape=(environment_space_length,)))
        # model.add(layers.Dense(24, activation="relu"))
        # model.add(layers.Dense(24, activation="relu"))
        # model.add(layers.Dense(action_space_length, activation="linear"))
        model.compile(loss="mse", optimizer=optimizer)  # type: ignore
    else:
        model = pretrained_model

    def update(state_index, action_index, next_state, reward: float = 0, done=False):
        target = model.predict(np.array([int(state_index)]))[0]
        if done:
            target[action_index] = reward
        else:
            target[action_index] = reward + gamma * np.max(
                model.predict(np.array([next_state]))
            )
        inputs = np.array([state_index])
        outputs = np.array([target])
        return model.fit(inputs, outputs, epochs=1, verbose="0", batch_size=1)

    def get_best_action(state, rand=True):
        if rand and np.random.rand() <= epsilon:
            # The agent acts randomly
            return random.randrange(action_space_length)

        print("in get_best_action")
        # Predict the reward value based on the given state
        act_values = model.predict(np.array([state]))

        # Pick the action based on the predicted reward
        action = np.argmax(act_values[0])
        return action

    scores = []
    for epoch in range(epoch_number):
        state = environment.reset()[0]  # Initial state
        stop = False
        score = 0
        steps = 0
        while not stop:
            steps += 1
            action = get_best_action(state)
            observation, reward, terminated, truncated, info = environment.step(action)
            score += reward
            stop = terminated or truncated
            if training_policy:
                update(state, action, observation, reward, stop)
            state = observation
            if stop:
                scores.append(score)
                break
            if epoch % 100 == 0:  # print log every 100 episode
                print(
                    "episode: {}/{}, moves: {}, score: {}".format(
                        epoch, epoch_number, steps, score
                    )
                )
    return model
