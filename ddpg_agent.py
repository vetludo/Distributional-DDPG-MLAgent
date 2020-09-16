import numpy as np
import random
from collections import namedtuple, deque

from .model import Actor, Critic

import torch
import torch.optim as optim

BUFFER_SIZE = int(5e5)  # replay buffer size
BATCH_SIZE = 128  # minibatch size
GAMMA = 0.99  # discount factor
TAU = 1e-3  # for soft update of target parameters
UPDATE_EVERY = 350
UPDATE_TYPE = 'hard'
LR_ACTOR = 5e-4  # learning rate of the actor
LR_CRITIC = 7.5e-4  # learning rate of the critic
WEIGHT_DECAY = 0  # L2 weight decay
E_START = 0.2
E_MIN = 0.03
E_DECAY = 0.999
V_MIN = 0
V_MAX = 0.3
NUM_ATOMS = 100

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DDPG_Agent:
    def __init__(self, state_size, action_size, random_seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.atoms = torch.linspace(V_MIN, V_MAX, NUM_ATOMS).to(device)
        self.t_step = 0

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.hard_update(self.actor_local, self.actor_target)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, NUM_ATOMS, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, NUM_ATOMS, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.hard_update(self.critic_local, self.critic_target)

        # Noise process
        self.e = E_START

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)

    def step(self, states, actions, rewards, next_states, dones):
        # Save experience / reward
        experiences = list(zip(states, actions, rewards, next_states, dones))
        for e in experiences:
            self.memory.add(*e)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences)

        self.t_step += 1

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            noise = self.gauss_noise(action.shape)
            action += noise
        return np.clip(action, -1, 1)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        atoms = self.atoms.unsqueeze(0)
        target_dist = self._get_targets(rewards, next_states).detach()

        log_probs = self.critic_local(states, actions, log=True)
        critic_loss = -(target_dist * log_probs).sum(-1).mean()

        predicted_action = self.actor_local(states)
        probs = self.critic_local(states, predicted_action)

        expected_reward = (probs * atoms).sum(-1)

        actor_loss = -expected_reward.mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if UPDATE_TYPE == 'soft':
            self.soft_update(self.critic_local, self.critic_target, TAU)
            self.soft_update(self.actor_local, self.actor_target, TAU)
        elif UPDATE_TYPE == 'hard':
            if self.t_step % UPDATE_EVERY == 0:
                self.hard_update(self.critic_local, self.critic_target)
                self.hard_update(self.actor_local, self.actor_target)

    def _get_targets(self, rewards, next_states):
        target_actions = self.actor_target(next_states)
        target_probs = self.critic_target(next_states, target_actions)

        projected_probs = self._categorical(rewards, target_probs)
        return projected_probs

    def _categorical(self, rewards, probs):
        vmin = V_MIN
        vmax = V_MAX
        atoms = self.atoms
        num_atoms = NUM_ATOMS
        gamma = GAMMA

        delta_z = (vmax - vmin) / (num_atoms - 1)

        projected_atoms = rewards + gamma * atoms.unsqueeze(0)
        projected_atoms.clamp_(vmin, vmax)
        b = (projected_atoms - vmin) / delta_z

        precision = 1
        b = torch.round(b * 10 ** precision) / 10 ** precision
        lower_bound = b.floor()
        upper_bound = b.ceil()

        m_lower = (upper_bound + (lower_bound == upper_bound).float() - b) * probs
        m_upper = (b - lower_bound) * probs

        projected_probs = torch.tensor(np.zeros(probs.size())).to(device)
        for idx in range(probs.size(0)):
            projected_probs[idx].index_add_(0, lower_bound[idx].long(), m_lower[idx].double())
            projected_probs[idx].index_add_(0, upper_bound[idx].long(), m_upper[idx].double())
        return projected_probs.float()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def hard_update(self, local_model, target_model):
        target_model.load_state_dict(local_model.state_dict())

    def gauss_noise(self, shape):
        self.e = max(E_MIN, self.e * E_DECAY)
        n = np.random.normal(0, 1, shape)
        return self.e * n


class ReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)
