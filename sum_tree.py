"""This class contains an implementation of Sum Tree, used for PER."""
import math
import random

import numpy as np


class SumTree:
    """Sum Tree.

    Binary tree in which the value of each node is the sum of its childs.
    Elements are not actually stored in the tree, only the indices can be sample,
    which correspond to the indices in the replay buffer.
    """

    def __init__(self, depth):
        """Initialize.

        Args:
            depth (int): Depth of the tree.
        """
        self.depth = depth
        self.tree = [np.zeros(2 ** i) for i in range(self.depth + 1)]

    def add(self, priority, index):
        """Add an element.

        Args:
            priority (float): Priority of the element.
            index (Union(int, List(int)): Index of the added element.
        """
        self.tree[-1][index] = priority
        self._propagate([index])

    def get(self, index):
        """Get element(s) at index."""
        return self.tree[-1][index]

    def sample(self, size):
        """Draw a sample respecting priorities."""
        p_total = self.tree[0][0]
        segment_size = p_total / size
        segments = [(k * segment_size, (k + 1) * segment_size) for k in range(size)]

        indices = []
        for min, max in segments:
            p = random.uniform(min, max)
            indices.append(self._retrieve_index(p))
        return indices

    def update(self, indices, priorities):
        """Update priorities of elements at indices."""
        for i, p in zip(indices, priorities):
            self.tree[-1][i] = p
        self._propagate(indices)

    @property
    def total_sum(self):
        """Sum of all elements."""
        return self.tree[0][0]

    def _propagate(self, indices):
        """Propagate sum from leaf at indices."""
        for d in reversed(range(self.depth)):
            indices = set([i // 2 for i in indices])
            for index in indices:
                self.tree[d][index] = (
                    self.tree[d + 1][2 * index] + self.tree[d + 1][2 * index + 1]
                )

    def _retrieve_index(self, p):
        """Retrieve the index of element with priority p."""
        i = 0
        for d in range(self.depth):
            left = self.tree[d + 1][2 * i]
            if p < left:
                i *= 2
            else:
                i = i * 2 + 1
                p -= left
        return i
