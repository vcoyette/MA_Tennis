# Project 2: Continuous Control

The environment for each agent is described as Markov Decision Process $(S, A, R, p, \gamma)$, which dynamics is unknown.
The action space $A$ is continuous.
The goal is to approximate the optimal policy $\pi^*$ maximizing the maximum upcoming cumulative reward in every state between the two agents.

## The algorithm

### DDPG

The algorithm used to solve this environment is [DDPG](https://arxiv.org/abs/1509.02971). 

As in DQN, the $q$ value is approximated by a neural network.
However, in order to bypass the optimization of $q$ to find the maximizing action, the selection of an action from a state is done by another neural network.
Let $\mu$ be this new network.
These two networks fit in the Actor-Critic framework.

As in DQN, target networks are defined to get a fixed target independant of the parameters in the TD error.
Thus, four network are define: $Q$, $Q'$, $\mu$ and $\mu'$.

The critic $Q$ is the network estimating the $q$ value.
Its loss is defined as the mean squared error in the Bellman optimality equation.
$$
\mathcal{L}_{critic}(\theta^Q) = \mathbb{E}_{s, a, r, s'} \big[((r + \gamma Q(s', \mu'(s'| \theta^{\mu'})| \theta^{Q'}) - Q(s, a| \theta^Q))^2\big]
$$

The Actor loss is defined as the $q$ value:
$$
\mathcal{L}_{actor}(\theta^{\mu}) = - \mathbb{E}_{s} Q(s, \mu(s | \theta^\mu) | \theta^Q)
$$

The target networks are updated every timesteps with a momentum $\tau$.

A replay buffer is used to simulate i.i.d. sampling to compute empirical gradient.

## Implementation

The actor network is composed of 3 fully connected layers, with respectively 128, 128 and 64 neurons each, with a batch normalisation layer between the first two fully connected.
The actor optimizer is Adam with a learning rate of $10^{-4}$.

The critic network is composed of 3 fully connected layers, with respectively 128, 128 and 64 neurons each, with a batch normalisation layer between the first two fully connected.
The output of the first layer and the action are concatenated before flowing into the second layer.
The critic optimizer is Adam with a learning rate of $10^{-3}$.

The discount factor $\gamma$ is equal to $0.99$. 
The target parameters $\theta^{Q'}$ and $\theta^{\mu'}$ are updated towards $\theta^Q$ and $\theta^\mu$ with a momentum $\tau$ of $10^-3$.

Instead of the Ornstein-Uhlenbeck noise process, a Gaussian noise with a standard deviation of 0.5 has been used on the actions to encourage exploration.

The training has been run with and without prioritizing the replay buffer.

The same actor network is controlling the 2 agents.
At each time step, two samples are added to the replay buffer (one for each agent).
One step of gradient of optimization is then ran.

## Results

The results (plot and number of episodes to solve) can be seen in the `Tennis.ipynb` notebook.
The non-prioritized replay buffer performes better, solving the environmnet in 1555 episodes.

The prioritized agent started better than the non-prioritize one, but finally did not solve the environment.
This may be done to bad hyperparameter choice, for example $\beta$ which was taken from the previous project while more episodes are ran, which means $\beta$ is increasing very fast to 1.

The training were carried out for 2000 episodes, and the actor weights for each configuration are provided in the repository.

## What next ?
To improve the performances of the model, some hyperparameters tuning may be performed.
Especially, the exploration noise may require some tweaking instead of remaining fixed to an arbirtary chosen value.
Moreover, some method such as MADDPG could be used to address the non-stationarity of both agent environment.
