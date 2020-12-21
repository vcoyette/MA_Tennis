"""This file contains configuration objects."""


class DDPGConfig:
    """DDPG configuration"""

    BUFFER_SIZE = int(1e5)  # replay buffer size
    BATCH_SIZE = 64  # minibatch size
    GAMMA = 0.99  # discount factor
    TAU = 1e-3  # for soft update of target parameters
    LR_ACTOR = 1e-4  # learning rate
    LR_CRITIC = 1e-3  # learning rate
    UPDATE_EVERY = 1  # how often to update the network
    UPDATE_STEPS = 1  # how often to update the network
    EXPL_NOISE = 0.5  # Exploration noise
    PER = False  # Use Prioritize Experience Replay
    SEED = 0  # Fix seed


class DDPGPERConfig(DDPGConfig):
    """DDPG configuration with Prioritize Experience Replay."""

    PER = True
