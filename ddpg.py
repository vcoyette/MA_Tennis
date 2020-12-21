"""Implementation of Deep Deterministic Policy Gradients (DDPG).

Paper: https://arxiv.org/abs/1509.02971
"""
import logging
import os.path as osp
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models import Actor, Critic
from replay_buffer import PrioritizeReplayBuffer, ReplayBuffer

# Use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DDPG:
    """Implementation of DDPG.

    This implementation is adapted to this particular environment running several agent.
    At each time step, the same actor is controlling each agent sequentially.
    """

    def __init__(self, state_size, action_size, config):
        """Initialize algorithm."""
        if config.PER:
            self.memory = PrioritizeReplayBuffer(
                config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED
            )
        else:
            self.memory = ReplayBuffer(
                config.BUFFER_SIZE, config.BATCH_SIZE, config.SEED
            )

        # Randomly initialize critic netowrk and actor
        self.actor = Actor(state_size, action_size, config.SEED).to(device)
        self.critic = Critic(state_size, action_size, config.SEED).to(device)

        # Initialize target networks with weights from actor critic
        # Actor
        self.actor_target = Actor(state_size, action_size, config.SEED).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # Critic
        self.critic_target = Critic(state_size, action_size, config.SEED).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Actor optimizer
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=config.LR_ACTOR
        )
        # Critic optimizer
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=config.LR_CRITIC
        )

        self.config = config

        self.t_step = 0

        self.expl_noise = config.EXPL_NOISE

    def step(self, target_sample=None, **kwargs):
        """Run a step of algorithm update."""
        # Sample a random minibatch of transitions
        states, actions, rewards, next_states, dones = self._draw_minibatch()

        # Compute the target Q value
        target_Q = self.critic_target(
            next_states, self.actor_target(next_states)
        ).detach()
        y = rewards + (1 - dones) * self.config.GAMMA * target_Q

        # Update critic by minimizing the loss
        current_Q = self.critic(states, actions)

        # Compute TD error
        td_error = y - current_Q

        if self.config.PER:
            # Get importance_sampling_weights
            weights = torch.Tensor(self.memory.importance_sampling()).unsqueeze(1)
            # Update priorities
            self.memory.update_priorities(td_error.detach().cpu().numpy())
            # Compute critic loss
            critic_loss = torch.mean(weights * td_error ** 2)
        else:
            # Compute critic loss
            critic_loss = torch.mean(td_error ** 2)

        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Clip gradient
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

        # Update the actor policy using the sampled policy gradient:
        actor_loss = -self.critic(states, self.actor(states)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # CLip gradient
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

        # Update target networks
        self.soft_update()

    def train(self, env, num_episode):
        """Train a DDPG agent."""
        scores = []
        scores_window = deque(maxlen=100)

        for episode in range(num_episode):
            # Init state and episode score
            states = env.reset(train_mode=True)
            score = np.zeros(states.shape[0])
            done = False

            # Run episode
            while not done:
                # Select and run action
                actions = self.predict_actions(states)
                # TODO: dynamic low and high selection
                actions = self.add_gaussian_noise(actions, -1, 1)
                next_states, rewards, dones = env.step(actions)

                # Store all n_agent episodes in replay buffer
                for state, action, reward, next_state, done in zip(
                    states, actions, rewards, next_states, dones
                ):
                    self.memory.add(state, action, reward, next_state, done)

                # Update time step
                self.t_step = (self.t_step + 1) % self.config.UPDATE_EVERY

                # Optimisation step if UPDATE_EVERY and enough examples in memory
                if self.t_step == 0 and len(self.memory) > self.config.BATCH_SIZE:
                    for _ in range(self.config.UPDATE_STEPS):
                        self.step()

                # Update state and scores
                states = next_states
                score += rewards

                # End episode if any of the agent is done, to avoid storing too much
                # Done transitions in the replay buffer
                done = any(dones)

            # Keep track of running mean
            scores_window.append(max(score))

            # Append current mean to scores list
            scores.append(np.mean(scores_window))

            # Logging
            print(
                "\rEpisode {}\tAverage Score: {:.2f}, Last Score: {:.2f}".format(
                    episode, np.mean(scores_window), max(score)
                ),
                end="",
            )
            if (episode + 1) % 100 == 0:
                print(
                    "\rEpisode {}\tAverage Score: {:.2f}".format(
                        episode, np.mean(scores_window)
                    )
                )

        return scores

    def soft_update(self):
        """Update the frozen target models."""
        tau = self.config.TAU
        # Actor
        for param, target_param in zip(
            self.critic.parameters(), self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

        # Critic
        for param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    def predict_actions(self, states, **kwargs):
        """Predict next actions based on current policy."""
        states = torch.from_numpy(states).float().unsqueeze(0).to(device)

        # Set actor to eval mode
        self.actor.eval()

        actions = []
        with torch.no_grad():
            for state in states:
                action = self.actor(state)
                actions.append(action.detach().numpy())

        # Set actor to train mode
        self.actor.train()

        return np.array(actions).squeeze()

    def add_gaussian_noise(self, action, low, high):
        """Add Gaussian noise to action, and clip between low and high."""
        return (action + np.random.normal(0, self.expl_noise, size=action.shape)).clip(
            low, high
        )

    def _draw_minibatch(self):
        """Draw a minibatch in the replay buffer."""
        states, actions, rewards, next_states, done = zip(*self.memory.sample())

        states = torch.Tensor(states).to(device)
        actions = torch.Tensor(actions).to(device)
        rewards = torch.Tensor(rewards).unsqueeze(1).to(device)
        next_states = torch.Tensor(next_states).to(device)
        done = torch.Tensor(done).unsqueeze(1).to(device)

        return states, actions, rewards, next_states, done

    def save_model(self, path, **kwargs):
        """Save actor model weights."""
        torch.save(self.actor.state_dict(), path)
