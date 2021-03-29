# TODO
# fix good shots tracking
# fix print-solved-in
# adaptive random decay based on good-shots
import argparse
import math
import os
import time
from collections import deque

import numpy as np
import torch

from logger import Logger
from maddpg import MADDPGAgent
from unity_environment import Env
import matplotlib.pyplot as plt

EPISODE_COUNT = 5000
MAX_T = 2000
SOLVE_SCORE = 0.5
CONTINUE_AFTER_SOLVE = True
ACT_PERCENT_RANDOMNESS = 0.90
RANDOMNESS_DECAY = 0.9977
# RANDOMNESS_DECAY_AFTER = 1000
RANDOMNESS_SHOTS_HURDLE = 0.2

def train(env, agent):
    log = Logger()
    scores = []
    avg_scores = []
    last_max = -math.inf
    randomness = ACT_PERCENT_RANDOMNESS
    solved_episode = None
    good_shots_last_100e = deque(maxlen=100)

    for e in range(1, EPISODE_COUNT):
        good_shots = 0
        rewards = []
        state, _, _ = env.reset()

        for t in range(MAX_T):
            action = agent.act(state, randomness)
            # Complete exploration for first 1000 episodes to get ball over net
            # Thereafter decay with each episode

            reward, next_state, done = env.transition(action)

            if any(r > 0.0 for r in reward):
                good_shots += 1

            agent.step(state, action, reward, next_state, done)

            state = next_state
            rewards.append(reward)
            if any(done):
                break

        # after every episode...
        scores.append(sum(np.array(rewards).sum(1)))
        np.save('log/scores', scores)
        avg_scores.append(sum(scores[-100:]) / 100)
        good_shots_last_100e.append(good_shots)

        # if e > RANDOMNESS_DECAY_AFTER:
        if np.mean(good_shots_last_100e) > RANDOMNESS_SHOTS_HURDLE:
            randomness = randomness * RANDOMNESS_DECAY
        elif randomness < ACT_PERCENT_RANDOMNESS:
            randomness = randomness / RANDOMNESS_DECAY

        print(
            f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | '
            f'Best average: {max(avg_scores):10.4f} | '
            f'Last score: {scores[-1]:10.4f} | '
            f'Good shots: {good_shots} | '
            f'Randomness: {randomness:.3f}',
            end='\r')

        if avg_scores[-1] > last_max:
            save(agent, f'/by_score/score_{avg_scores[-1]:.2f}')
            last_max = avg_scores[-1]

        if avg_scores[-1] > SOLVE_SCORE and e > 100:
            if solved_episode is not None:
                print(
                    f'\rSolved after {e} episodes with avg. score of {avg_scores:10.4f}')
            if not CONTINUE_AFTER_SOLVE:
                break

        if e % 100 == 0:
            log.write_line(
                f'E: {e:6} | Average: {avg_scores[-1]:10.4f} | '
                f'Best average: {max(avg_scores):10.4f} | '
                f'Last score: {scores[-1]:10.4f} | '
                f'Good shots avg: {np.mean(good_shots_last_100e)} | '
                f'Randomness: {randomness:.3f}')


def save(agent, tag):
    for i, to_save in enumerate(agent.agents):
        torch.save(to_save.actor_local.state_dict(),
                   os.getcwd() + f"/models/{tag}_actor_{i}.weights")
        torch.save(to_save.critic_local.state_dict(),
                   os.getcwd() + f"/models/{tag}_critic_{i}.weights")


def play(env, agent, playthrougs=10):
    for e in range(playthrougs):
        state, _, _ = env.reset(train_mode=False)
        is_done = False

        total_max_reward = 0
        while not is_done:
            reward, n_state, done = env.transition(agent.act(state, random=None))
            state = n_state
            total_max_reward = np.array(reward).max()
            is_done = any(done)

        print(f'P:{e}   Score:{total_max_reward}')


def plot_scores(path):
    losses = np.genfromtxt(path)
    losses = losses[~np.isnan(losses)].reshape(-1, 5)
    losses = np.array(losses)

    fig, axs = plt.subplots(1, 2)
    for i in range(2):
        axs[i].plot(losses.T[0], losses.T[1], label='Avg. Score', alpha=0.5)
        axs[i].plot(losses.T[0], losses.T[2], label='Max Avg. Score', alpha=0.5)
        axs[i].plot(losses.T[0], [0.5] * len(losses), label='Target')
        if i == 0:
            axs[i].set(yscale='log', xlabel='Episode #', ylabel='Score (log scale)')
        else:
            axs[i].set(xlabel='Episode #', ylabel='Score')
        axs[i].legend()

    fig.suptitle("Average Score for both agents over 100 episodes")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'play', 'plot'])
    parser.add_argument('--env', type=str, default='Tennis.app')
    parser.add_argument('--mock', type=bool, default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--seed', type=int, default=round(time.time()))
    parser.add_argument('--graphics', type=bool, default=False)
    args = parser.parse_args()

    if args.mode == 'plot':
        print('Plotting scores...\n')
        plot_scores(args.load)
    else:

        print('Loading environment and instantiating agent...\n')
        env = Env(args.env, is_mock=args.mock, seed=args.seed, no_graphics=args.graphics)
        try:
            agent = MADDPGAgent(args.seed, checkpoint_filename=args.load)

            if args.mode == 'play':
                print("Running in evaluation mode...\n")
                play(env, agent)
            else:
                print("Training agent...\n")
                try:
                    train(env, agent)
                finally:
                    print('\n')
                    save(agent, 'final')
        finally:
            env.close()
