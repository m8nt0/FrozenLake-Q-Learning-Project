# FrozenLake-Q-Learning-Project
demonstrates how to use Q-Learning to solve the FrozenLake environment from OpenAI Gym. It includes advanced features such as logging, configuration files, model saving/loading, plotting, and hyperparameter tuning.

## Requirements

- Python 3.6+
- numpy
- gym
- matplotlib
- scikit-optimize
- pyyaml

## Project Structure

frozenlake_rl_project/
│
├── configs/
│   ├── config.json
│   └── params.json
├── logs/
│   ├── training_rewards.npy
│   ├── training_lengths.npy
│   └── training.log
├── models/
│   └── q_table.npy
├── plots/
│   └── training_plot.png
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── q_learning_agent.py
│   ├── trainer.py
│   ├── evaluator.py
│   └── utils.py
├── README.md
└── requirements.txt
