import json
import yaml
import numpy as np
import matplotlib.pyplot as plt

def load_config(file_path):
    with open(file_path, 'r') as file:
        if file_path.endswith('.json'):
            return json.load(file)
        elif file_path.endswith('.yaml') or file_path.endswith('.yml'):
            return yaml.safe_load(file)
        else:
            raise ValueError('Unsupported file format.')

def save_q_table(q_table, file_path):
    np.save(file_path, q_table)

def load_q_table(file_path):
    return np.load(file_path)

def plot_rewards(rewards, lengths, plot_file):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Rewards')

    plt.subplot(1, 2, 2)
    plt.plot(lengths)
    plt.xlabel('Episode')
    plt.ylabel('Episode Length')
    plt.title('Training Lengths')

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.show()
