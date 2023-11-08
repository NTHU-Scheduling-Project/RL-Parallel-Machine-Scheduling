import os
import sys
import math
import random
#import matplotlib
#import matplotlib.pyplot as plt
import collections
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from util.replay_buffer import ReplayBuffer
from model.cnn_model import Net

Transition = collections.namedtuple(
    'Transition',
    'state action reward next_state is_final_step legal_actions_mask')

ILLEGAL_ACTION_LOGITS_PENALTY = sys.float_info.min

class DQN:
    """DQN Agent implementation in PyTorch"""
    def __init__(
        self,
        n_actions,
        replay_buffer_capacity: int = 10000,
        batch_size: int = 128,
        learning_rate: float = 0.001,
        tau: float = 0.005,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_duration: int = 1000
    ):
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_duration = epsilon_decay_duration
        self.epsilon_threshold = 1.0

        self.replay_buffer = ReplayBuffer(replay_buffer_capacity)
        self.prev_time_step = None
        self.prev_action = None

        self.step_counter = 0

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.q_network = Net(n_actions, dueling=True).to(self.device)
        self.target_q_network = Net(n_actions, dueling=True).to(self.device)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=learning_rate, amsgrad=True)
        
        # Statistics
        self.q_record: collections.deque = collections.deque(maxlen=1000)
        self.loss_record: collections.deque = collections.deque(maxlen=100)

    def add_transition(self, state, action, reward, next_state, is_final_step, legal_actions):
        legal_actions_mask = np.zeros(self.n_actions)
        legal_actions_mask[legal_actions] = 1.0
        transition = Transition(
            state = state,
            action = action,
            reward = reward,
            next_state = next_state,
            is_final_step = is_final_step,
            legal_actions_mask = legal_actions_mask
        )
        self.replay_buffer.add(transition)

    def select_action(self, state, legal_actions, flag = "training"):
        sample = random.random()
        self.epsilon_threshold = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(-1. * self.step_counter / self.epsilon_decay_duration)
        self.step_counter += 1

        #if sample < self.epsilon_threshold and flag != "val":
        if sample < self.epsilon_threshold or flag != "val":        
            action = np.random.choice(legal_actions)
        else:
            q_values = self.q_network(state).detach()[0]
            legal_q_values = q_values[legal_actions]
            action = legal_actions[torch.argmax(legal_q_values)]
        
        return action
    
    def learn(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        transitions = self.replay_buffer.sample(self.batch_size)
        states = torch.Tensor(np.array([t.state for t in transitions])).to(self.device)
        actions = torch.LongTensor([t.action for t in transitions]).to(self.device)
        rewards = torch.Tensor([t.reward for t in transitions]).to(self.device)
        next_states = torch.Tensor(np.array([t.next_state for t in transitions])).to(self.device)
        are_final_steps = torch.Tensor([t.is_final_step for t in transitions]).to(self.device)
        legal_actions_mask = torch.Tensor(np.array([t.legal_actions_mask for t in transitions])).to(self.device)

        q_values = self.q_network(states)
        target_q_values = self.target_q_network(next_states).detach()

        self.q_record.extend(q_values.detach().cpu().numpy().ravel())

        illegal_actions_mask = 1 - legal_actions_mask
        legal_target_q_values = target_q_values.masked_fill(illegal_actions_mask.bool(), ILLEGAL_ACTION_LOGITS_PENALTY)
        max_next_q = torch.max(legal_target_q_values, dim=1)[0]
        
        target = (rewards + (1 - are_final_steps) * self.gamma * max_next_q)
        action_indices = torch.stack([
            torch.arange(q_values.shape[0], dtype=torch.long).to(self.device), actions], dim=0)
        
        predictions = q_values[list(action_indices)]

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(predictions, target)

        self.loss_record.append(float(loss.detach().cpu().numpy()))

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_value_(self.q_network.parameters(), 100)
        self.optimizer.step()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_q_network.state_dict()
        policy_net_state_dict = self.q_network.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*self.tau + target_net_state_dict[key]*(1-self.tau)
        self.target_q_network.load_state_dict(target_net_state_dict)

    def _mean_or_nan(self, xs):
        """Return its mean a non-empty sequence, numpy.nan for a empty one."""
        return np.mean(xs) if xs else np.nan

    def get_statistics(self):
        return [
            ("average_q", self._mean_or_nan(self.q_record)),
            ("average_loss", self._mean_or_nan(self.loss_record)),
        ]

    def save(self, data_path, optimizer_data_path=None):
        """Save checkpoint/trained model and optimizer.

        Args:
        data_path: Path for saving model. It can be relative or absolute but the
            filename should be included. For example: q_network.pt or
            /path/to/q_network.pt
        optimizer_data_path: Path for saving the optimizer states. It can be
            relative or absolute but the filename should be included. For example:
            optimizer.pt or /path/to/optimizer.pt
        """
        if not os.path.exists(data_path):
            os.makedirs(data_path)

        torch.save(self.q_network, data_path + "/q_network.pt")
        torch.save(self.target_q_network, data_path + "/target_q_network.pt")
        #if optimizer_data_path is not None:
        torch.save(self.optimizer, data_path + "/optimizer.pt")

    def load(self, data_path, optimizer_data_path=None):
        """Load checkpoint/trained model and optimizer.

        Args:
            data_path: Path for loading model. It can be relative or absolute but the
            filename should be included. For example: q_network.pt or
            /path/to/q_network.pt
            optimizer_data_path: Path for loading the optimizer states. It can be
            relative or absolute but the filename should be included. For example:
            optimizer.pt or /path/to/optimizer.pt
        """
        self.q_network = torch.load(data_path + "/q_network.pt")
        self.target_q_network = torch.load(data_path + "/target_q_network.pt")
        #if optimizer_data_path is not None:
        self.optimizer = torch.load(data_path + "/optimizer.pt")