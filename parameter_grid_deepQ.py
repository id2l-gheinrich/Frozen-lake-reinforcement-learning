import random
import gymnasium as gym
import numpy as np
import json

from src.Algorithms.Deep_Q import Deep_Q

from src.Classes.Policy import Policy
from src.Classes.Agent import Agent
from src.Functions.run import run_static

#    ____             _                     _____             __ _                       _   _
#   |  _ \           | |                   / ____|           / _(_)                     | | (_)
#   | |_) | __ _  ___| | ___   _ _ __     | |     ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __
#   |  _ < / _` |/ __| |/ / | | | '_ \    | |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \
#   | |_) | (_| | (__|   <| |_| | |_) |   | |___| (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | |
#   |____/ \__,_|\___|_|\_\\__,_| .__/     \_____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
#                               | |                                 __/ |
#                               |_|                                |___/

output_file = "data/output_deep.json"

#     _____      _     _       _____             __ _                       _   _
#    / ____|    (_)   | |     / ____|           / _(_)                     | | (_)
#   | |  __ _ __ _  __| |    | |     ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __
#   | | |_ | '__| |/ _` |    | |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \
#   | |__| | |  | | (_| |    | |___| (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | |
#    \_____|_|  |_|\__,_|     \_____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
#                                                      __/ |
#                                                     |___/
test_epoch = 100
grid_steps = 2
mail = {
    "epoch_number": {
        "start": 2000,
        "end": 8000,
        "steps": grid_steps,
    },
    "epsilon": {
        "start": 0.5,
        "end": 0.98,
        "steps": grid_steps,
    },
    "alpha": {
        "start": 0.1,
        "end": 0.98,
        "steps": grid_steps,
    },
    "gamma": {
        "start": 0.8,
        "end": 0.98,
        "steps": grid_steps,
    },
}
seed = 0

#    ______            _                                      _         _____             __ _                       _   _
#   |  ____|          (_)                                    | |       / ____|           / _(_)                     | | (_)
#   | |__   _ ____   ___ _ __ ___  _ __  _ __ ___   ___ _ __ | |_     | |     ___  _ __ | |_ _  __ _ _   _ _ __ __ _| |_ _  ___  _ __
#   |  __| | '_ \ \ / / | '__/ _ \| '_ \| '_ ` _ \ / _ \ '_ \| __|    | |    / _ \| '_ \|  _| |/ _` | | | | '__/ _` | __| |/ _ \| '_ \
#   | |____| | | \ V /| | | | (_) | | | | | | | | |  __/ | | | |_     | |___| (_) | | | | | | | (_| | |_| | | | (_| | |_| | (_) | | | |
#   |______|_| |_|\_/ |_|_|  \___/|_| |_|_| |_| |_|\___|_| |_|\__|     \_____\___/|_| |_|_| |_|\__, |\__,_|_|  \__,_|\__|_|\___/|_| |_|
#                                                                                               __/ |
#                                                                                              |___/
desc = ["SFFF", "FHFH", "FFFH", "HFFG"]  # Same as the map called "4*4"
environment = gym.make(
    "FrozenLake-v1", desc=desc, is_slippery=True, render_mode="rgb_array"
)

#    __  __       _           _
#   |  \/  |     (_)         | |
#   | \  / | __ _ _ _ __     | |     ___   ___  _ __
#   | |\/| |/ _` | | '_ \    | |    / _ \ / _ \| '_ \
#   | |  | | (_| | | | | |   | |___| (_) | (_) | |_) |
#   |_|  |_|\__,_|_|_| |_|   |______\___/ \___/| .__/
#                                              | |
#                                              |_|


result = []
cpt = 0
for epoch_number in np.linspace(
    mail["epoch_number"]["start"],
    mail["epoch_number"]["end"],
    mail["epoch_number"]["steps"],
):
    for epsilon in np.linspace(
        mail["epsilon"]["start"], mail["epsilon"]["end"], mail["epsilon"]["steps"]
    ):
        for alpha in np.linspace(
            mail["alpha"]["start"], mail["alpha"]["end"], mail["alpha"]["steps"]
        ):
            for gamma in np.linspace(
                mail["gamma"]["start"], mail["gamma"]["end"], mail["gamma"]["steps"]
            ):
                options = {
                    "epoch_number": int(epoch_number),
                    "epsilon": epsilon,
                    "alpha": alpha,
                    "gamma": gamma,
                }

                random.seed(seed)
                environment.reset(seed=seed)
                pre_trained_model = Deep_Q(environment, **options)
                print("Done training")
                # TODO utiliser proprement le modèle appris
                deterministic_policy = Policy.buildOptimalPolicyFromNeuralNetwork(
                    pre_trained_model
                )
                success = 0
                random.seed(seed)
                environment.reset(seed=seed)
                for epoch in range(test_epoch):
                    test_agent = Agent(deterministic_policy)
                    run_static(environment=environment, agent=test_agent)
                    success += test_agent.current_state_index == (
                        environment.observation_space.n - 1  # type: ignore
                    )
                result.append(
                    {
                        "algorithm": Deep_Q.__name__,
                        "success": success / test_epoch,
                        "epoch_number": epoch_number,
                        "epsilon": epsilon,
                        "alpha": alpha,
                        "gamma": gamma,
                    }
                )

                with open(output_file, "w") as the_file:
                    # Serializing json
                    # Globally, may be faster here
                    # See https://stackoverflow.com/a/57087055
                    json_object = json.dumps(result, indent=4)
                    the_file.write(json_object)
                cpt += 1
                print(
                    str(cpt)
                    + "/"
                    + str(
                        mail["alpha"]["steps"]
                        * mail["gamma"]["steps"]
                        * mail["epsilon"]["steps"]
                        * mail["epoch_number"]["steps"]
                        * 2
                    )
                )
