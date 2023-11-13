# RL-Parallel-Machine-Scheduling
This is the implementation of the paper [Deep Reinforcement Learning for Minimizing Tardiness in Parallel Machine Scheduling With Sequence Dependent Family Setups](https://ieeexplore.ieee.org/document/9486959). 

We modified the objective to maximize the average utilization of all machines.

Two versions are implemented:
- the `main` branch version can be used to train on a single instance.
- the `training_dataset` branch version can be used to train on a dataset containing multiple instances.

## How to use
- `data` contains the data we used to train the model.
- `main.py` is the entry point of the project.
- `eny.py` is the PMSP environment built on OpenAI Gym.
- `dqn.py` is the implementation of a DQN agent, modified from [open_spiel](https://github.com/google-deepmind/open_spiel).
- `runner.py` is the runner of the training/validation function.

One can run the following code to train the agent:
```python
python3 main.py --gpu=0
```
To test the trained model:
```python
python3 main.py --gpu=0 --test