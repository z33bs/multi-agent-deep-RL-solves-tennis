from unity_environment import Env
import numpy as np

# reset the environment
env = Env('Tennis.app')
env.reset()
# number of agents 
num_agents = env.num_agents
print('Number of agents:', num_agents)

# size of each action
action_size = env.action_size
print('Size of each action:', action_size)

# examine the state space 
state_size = env.state_size
print('There are {} agents. Each observes a state with length: {}'.format(num_agents, state_size))
print('The state for the first agent looks like:', env.state[0])

for i in range(1, 6):                                      # play game for 5 episodes
    env.reset(train_mode=False)                            # reset the environment
    states = env.state                                     # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)
    t = 0
    while True:
        t += 1
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        rewards, next_states, dones = env.transition(actions) # send all actions to tne environment
        scores += rewards                                   # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            print('Episode lasted {} steps'.format(t))
            break
    print('Score (max over agents) from episode {}: {}'.format(i, np.max(scores)))

env.close()
