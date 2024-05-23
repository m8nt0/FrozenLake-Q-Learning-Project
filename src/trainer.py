import numpy as np
import logging
from q_learning_agent import QLearningAgent
from utils import save_q_table, plot_rewards

def train(config, params):
    env = gym.make(config['env_name'], is_slippery=config['is_slippery'])
    agent = QLearningAgent(
        state_size=env.observation_space.n,
        action_size=env.action_space.n,
        alpha=params['alpha'],
        gamma=params['gamma'],
        epsilon=params['epsilon'],
        epsilon_min=params['epsilon_min'],
        epsilon_decay=params['epsilon_decay']
    )

    rewards = []
    lengths = []

    for episode in range(params['num_episodes']):
        state = env.reset()
        done = False
        step = 0
        total_reward = 0

        while not done and step < params['max_steps_per_episode']:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1

        rewards.append(total_reward)
        lengths.append(step)
        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            logging.info(f'Episode {episode + 1}/{params["num_episodes"]}, epsilon: {agent.epsilon:.2f}')

    save_q_table(agent.q_table, config['q_table_file'])
    np.save('logs/training_rewards.npy', rewards)
    np.save('logs/training_lengths.npy', lengths)
    plot_rewards(rewards, lengths, config['reward_plot'])
