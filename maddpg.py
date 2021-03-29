import os
import numpy as np
import torch
from buffer import ReplayBuffer
from ddpgagent import DDPGAgent

BUFFER_SIZE = int(1E5)
BATCH_SIZE = 128
UPDATE_FREQUENCY = 2
GAMMA = .99
NUM_AGENTS = 2
DEVICE = 'cpu'


class MADDPGAgent():
    def __init__(self, seed, checkpoint_filename=None):

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, DEVICE, seed)
        self.t = 0

        self.agents = [DDPGAgent(index, NUM_AGENTS, seed, DEVICE) for index in range(NUM_AGENTS)]

        if checkpoint_filename:
            for i, to_load in enumerate(self.agents):
                f"{os.getcwd()}/models/{checkpoint_filename}_actor_{i}.weights"
                actor_file = torch.load(f"{os.getcwd()}/models/{checkpoint_filename}_actor_{i}.weights"
                                        , map_location=DEVICE)
                critic_file = torch.load(f"{os.getcwd()}/models/{checkpoint_filename}_critic_{i}.weights"
                                         , map_location=DEVICE)
                to_load.actor_local.load_state_dict(actor_file)
                to_load.actor_target.load_state_dict(actor_file)
                to_load.critic_local.load_state_dict(critic_file)
                to_load.critic_target.load_state_dict(critic_file)
            print(f'Files loaded with prefix {checkpoint_filename}')

    def step(self, all_states, all_actions, all_rewards, all_next_states, all_dones):
        all_states = all_states.reshape(1, -1)
        all_next_states = all_next_states.reshape(1, -1)
        self.memory.add(all_states, all_actions, all_rewards, all_next_states, all_dones)
        self.t = (self.t + 1) % UPDATE_FREQUENCY
        if self.t == 0 and (len(self.memory) > BATCH_SIZE):
            experiences = [self.memory.sample() for _ in range(NUM_AGENTS)]
            self.learn(experiences, GAMMA)

    def act(self, all_states, random):
        all_actions = []
        for agent, state in zip(self.agents, all_states):
            action = agent.act(state, random=random)
            all_actions.append(action)
        return np.array(all_actions).reshape(1, -1)

    def learn(self, experiences, gamma):
        all_actions = []
        all_next_actions = []
        for i, agent in enumerate(self.agents):
            states, _, _, next_states, _ = experiences[i]
            agent_id = torch.tensor([i]).to(DEVICE)
            state = states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            next_state = next_states.reshape(-1, 2, 24).index_select(1, agent_id).squeeze(1)
            all_actions.append(agent.actor_local(state))
            all_next_actions.append(agent.actor_target(next_state))
        for i, agent in enumerate(self.agents):
            agent.learn(i, experiences[i], gamma, all_next_actions, all_actions)
