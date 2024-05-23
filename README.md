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

- `configs/`: Contains configuration files.
- `logs/`: Contains logs and saved data.
- `models/`: Contains saved models.
- `plots/`: Contains generated plots.
- `src/`: Contains source code.

## Setup

1. **Install the required packages:**
   ```bash
   pip install -r requirements.txt

## Instructions

### Configure the Hyperparameters

Configure the hyperparameters in `configs/params.json`.

### Run the Training Script

    ```bash
    python src/main.py --mode train

## Run the Evaluation Script

    ```bash
    python src/main.py --mode evaluate

## Configuration

- **configs/config.json**: Contains environment and file paths configurations.
- **configs/params.json**: Contains hyperparameters for the Q-Learning algorithm.

## Logging and Plotting

- Logs are saved in the `logs/` directory.
- Training progress plots are saved in the `plots/` directory.

## Advanced Features

- **Logging**: Logs training progress to a file.
- **Configuration Files**: Use JSON for environment and hyperparameter configurations.
- **Model Saving and Loading**: Save and load the Q-table to and from a file.
- **Plotting**: Generate plots for training rewards and episode lengths.
- **Hyperparameter Tuning**: Supports basic hyperparameter tuning using a configuration file.

## Example Usage

### Training the Agent

To train the Q-Learning agent, run:

    ```bash
    python src/main.py --mode train

## Evaluating the Agent

To evaluate the trained Q-Learning agent, run:

    ```bash
    python src/main.py --mode evaluate

