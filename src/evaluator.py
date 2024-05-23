import numpy as np
import gym
import logging
from q_learning_agent import QLearningAgent
from utils import load_q_table

def evaluate(config, params):
    env = gym.make(config['env_name'], is_slippery=config['is_slippery'])
    q_table = load_q_table(config['q_table_file'])
    agent = QLearningAgent(env.observation_space.n, env.action_space.n)
    agent.q_table = q_table

    total_rewards = 0
    total_lengths = 0
    success_count = 0

    for _ in range(100):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0

        while not done and step < params['max_steps_per_episode']:
            action = np.argmax(agent.q_table[state])
            next_state, reward, done, _ = env.step(action)
            state = next_state
            total_reward += reward
            step += 1

        total_rewards += total_reward
        total_lengths += step
        success_count += 1 if total_reward > 0 else 0

    avg_reward = total_rewards / 100
    avg_length = total_lengths / 100
    success_rate = success_count / 100

    logging.info(f'Average reward: {avg_reward}')
    logging.info(f'Average length: {avg_length}')
    logging.info(f'Success rate: {success_rate}')
    print(f'Average reward: {avg_reward}')
    print(f'Average length: {avg_length}')
    print(f'Success rate: {success_rate}')
