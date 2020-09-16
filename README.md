# Distributional DDPG mlagent
A example code of Distributional DDPG (Deep Deterministic Policy Gradient) for a simulation of robotic arms from Unity ML agent

## The Environment
For this project, I will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![reacher demo](/assets/reacher.gif)

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

## Solving the Environment

This version contains 20 identical agents, each with its own copy of the environment. After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores. This yields an **average score** for each episode (where the average is over all 20 agents). The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

## Installation

Install Unity ml-agents.
```
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/
```
Install the project requirements.
```
pip install -r requirements.txt
```
Run continous_control.ipynb and follow the instruction.

## Report

The agent is trained by Distributional DDPG for this environment. After episode 200, the agents has already got an average score +30 over 100 consecutive episodes.

Episode 100	Average Score: 7.59\
Episode 200	Average Score: 31.91\
Episode 300	Average Score: 36.51\
Episode 400	Average Score: 37.68\
Episode 500	Average Score: 37.54

![report](/assets/report.png)

## Future Work

- Implement Prioritized Experience Replay and n-Rollout trajectory to improve the performance and reduce the training time.
- Evaluates the performance of various deep RL algorithms such as REINFORCE, TNPG, RWR, REPS, TRPO, CEM, CMA-ES and D4PG, on this continuous control tasks to figure out which are best suited.

Stay tuned.
