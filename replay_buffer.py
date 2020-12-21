import math
import random
from collections import deque

import numpy as np
import torch

from sum_tree import SumTree


class ReplayBuffer:
    """Replay buffer to store experiences."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Args:
            buffer_size (int): Size of the buffer.
            batch_size (int): Size of minibatches to sample.
        """
        self.data = []
        self.buffer_size = buffer_size
        self.index = 0
        self.batch_size = batch_size
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add transition inside the Replay buffer."""
        if len(self.data) < self.buffer_size:
            self.data.append(None)

        self.data[self.index] = (state, action, reward, next_state, done)
        self.index = (self.index + 1) % self.buffer_size

    def sample(self):
        """Get sample."""
        return random.sample(self.data, k=self.batch_size)

    def __len__(self):
        """Current lenght of replay buffer."""
        return len(self.data)


class PrioritizeReplayBuffer(ReplayBuffer):
    """Prioritize experience replay."""

    def __init__(
        self,
        buffer_size,
        batch_size,
        seed,
        beta_start=0.4,
        delta_beta=1e-5,
        alpha=0.6,
        eps=1e-8,
    ):
        """Initialize PER.

        Args:
            buffer_size (int): Size of replay buffer. The actual size will be the
                first power of 2 greater than buffer_size.
            batch_size (int): Size of batches to draw.
            seed (float): Seed.
            beta_start (float): Initial value for beta (importance sampling exponent)
            delta_beta (float): Beta increment at each time step.
            alpha (float): Priority exponent.
            eps (float): Small positive number to avoid unsampling 0 prioritized examples.
        """
        # Depth of sum tree
        depth = int(math.log2(buffer_size)) + 1
        super(PrioritizeReplayBuffer, self).__init__(2 ** depth, batch_size, seed)

        # Initialize sum tree to keep track of the sum of priorities
        self.priorities = SumTree(depth)

        # Current max priority
        self.max_p = 1.0

        # PER Parameters
        self.alpha = alpha
        self.eps = eps
        self.beta = beta_start
        self.delta_beta = delta_beta

    def add(self, state, action, reward, next_state, done):
        """Add transition inside the Replay buffer."""
        # Add in the sum tree with current max priority
        self.priorities.add(self.max_p, self.index)
        super().add(state, action, reward, next_state, done)

    def sample(self):
        """Get sample."""
        # Get indices to sample from sum tree
        # Store these indices to compute importance sampling later
        self.last_indices = self.priorities.sample(self.batch_size)

        # Return transitions corresponding to this indices
        return [self.data[i] for i in self.last_indices]

    def update_priorities(self, td_error):
        """Update priorities."""
        # Compute new priorites
        new_priorities = (abs(td_error) + self.eps) ** self.alpha

        # Update sum tree
        self.priorities.update(self.last_indices, new_priorities)

        # Update the current max priority
        self.max_p = max(self.max_p, max(new_priorities))

    def importance_sampling(self):
        """Compute importance sampling weights of last sample."""
        # Get probabilities
        probs = self.priorities.get(self.last_indices) / self.priorities.total_sum

        # Compute weights
        weights = (len(self) * probs) ** (-self.beta)
        weights /= max(weights)

        # Update beta
        self.beta = min(self.beta + self.delta_beta, 1)

        # Return weights
        return weights
