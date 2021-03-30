import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from network import Network

ACTION_SIZE = 2
STATE_SIZE = 24
ACTOR_HIDDEN_DIMS = [256, 128]
CRITIC_HIDDEN_DIMS = [512, 256]
TAU = 1e-3  # For soft-updates of target
ACTOR_LR = 1e-3
CRITIC_LR = 1e-3


class DDPGAgent():
    def __init__(self, index, num_agents, seed, device):
        random.seed(seed)
        np.random.seed(seed)

        self.index = index
        self.device = device

        self.actor_local = Network(STATE_SIZE, ACTOR_HIDDEN_DIMS, ACTION_SIZE, torch.tanh, seed)
        self.actor_target = Network(STATE_SIZE, ACTOR_HIDDEN_DIMS, ACTION_SIZE, torch.tanh, seed)
        self.actor_optimizer = Adam(self.actor_local.parameters(), lr=ACTOR_LR)
        self.critic_local = Network(num_agents * (STATE_SIZE + ACTION_SIZE), CRITIC_HIDDEN_DIMS, 1, None, seed)
        self.critic_target = Network(num_agents * (STATE_SIZE + ACTION_SIZE), CRITIC_HIDDEN_DIMS, 1, None, seed)
        self.critic_optimizer = Adam(self.critic_local.parameters(), lr=CRITIC_LR, weight_decay=0)

    def act(self, state, random):
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(torch.from_numpy(state).float().to(self.device)).cpu().data.numpy()
        self.actor_local.train()
        if random is not None:
            action = (1 - random) * action + random * (np.random.rand(ACTION_SIZE) - 0.5) * 2.0
        return np.clip(action, -1, 1)

    def learn(self, index, experiences, gamma, all_next_actions, all_actions):
        states, actions, rewards, next_states, dones = experiences

        self.critic_optimizer.zero_grad()

        index = torch.tensor([index]).to(self.device)
        actions_next = torch.cat(all_next_actions, dim=1).to(self.device)
        with torch.no_grad():
            q_next = self.critic_target(self.critic_input(next_states, actions_next))
        q_exp = self.critic_local(self.critic_input(states, actions))
        q_t = rewards.index_select(1, index) + (gamma * q_next * (1 - dones.index_select(1, index)))
        F.mse_loss(q_exp, q_t.detach()).backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()

        actions_pred = [actions if i == self.index else actions.detach() for i, actions in enumerate(all_actions)]
        actions_pred = torch.cat(actions_pred, dim=1).to(self.device)
        actor_loss = -self.critic_local(self.critic_input(states, actions_pred)).mean()
        actor_loss.backward()

        self.actor_optimizer.step()

        self.actor_target.soft_update(self.actor_local, TAU)
        self.critic_target.soft_update(self.critic_local, TAU)

    def critic_input(self, states, actions):
        return torch.cat((states, actions), dim=1)
