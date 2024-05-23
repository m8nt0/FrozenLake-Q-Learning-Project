import argparse
import logging
import json
import yaml
from q_learning_agent import QLearningAgent
from trainer import train
from evaluator import evaluate
from utils import load_config

def main():
    parser = argparse.ArgumentParser(description='Q-Learning for FrozenLake')
    parser.add_argument('--mode', choices=['train', 'evaluate'], required=True, help='Mode: train or evaluate')
    args = parser.parse_args()

    config = load_config('configs/config.json')
    with open('configs/params.json', 'r') as file:
        params = json.load(file)

    logging.basicConfig(filename=config['log_file'], level=logging.INFO, format='%(asctime)s - %(message)s')
    logging.info(f'Starting in {args.mode} mode')

    if args.mode == 'train':
        train(config, params)
    elif args.mode == 'evaluate':
        evaluate(config, params)

if __name__ == '__main__':
    main()
