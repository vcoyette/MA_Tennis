import matplotlib.pyplot as plt
import numpy as np
from unityagents import UnityEnvironment

from config import DDPGConfig
from ddpg import DDPG
from environment import UnityGymAdapter


def train():
    """Run training."""
    env = UnityEnvironment(file_name="Tennis_Linux/Tennis.x86_64")
    env = UnityGymAdapter(env)

    config = DDPGConfig()

    agent = DDPG(env.state_size, env.action_size, config)

    scores = agent.train(env, 2000)

    env.close()

    # plot the scores
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.show()


if __name__ == "__main__":
    train()
