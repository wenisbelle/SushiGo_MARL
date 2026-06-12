# SushiGo_MARL
Requirement for the Multi Agent Systems class

## Requirements 

- Docker

- Nvidia toolkit container

## Environment

Build the image:

    docker build -f dockerfile -t sushi_go .

Create the container:

    docker-compose up -d

    docker exec -it sushi_go_container bash

## Trainning the DQN 

Just run:

    python3 train/train_dqn.py --save-path models/DQN/NAME_OF_THE_MODEL.pt

## Testing

Just run:

    python3 test/play.py --model PATH_TO_YOUR_MODEL --n-players NUMBER_OF_PLAYER

## DeepCFR info

/models/DeepCFR folder has the trained algorithm for 2 and 3 players.

/test folder has some scripts for test DeepCFR against DQN, DeepCFR (second best trained model) and random. 

* All the testing scripts has a "--deepcfr-greedy" param:

    * In DeepCFR, the learned policy does not represent a single fixed action, but rather a probability distribution over the legal actions. For example, if the network assigns probabilities [0.10, 0.35, 0.55] for a given state, the agent in stochastic mode may select any of those actions according to their probabilities, so it will not always choose the highest-probability action. In contrast, when evaluating in greedy mode, the agent directly selects the action with the highest probability using argmax, turning the learned mixed strategy into a deterministic policy. Stochastic mode is more aligned with the theoretical idea of playing DeepCFR’s average strategy.

    * Both behaviors were tested. "Greedy" mode achieved the best result (winning 80% of 1000 games against DQN), but the stochastic mode has also a good performance (winning 40% of 1000 games against DQN). **Results for 2 players**