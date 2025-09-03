# Frac-Surg-Gym
An environment for training a RL agent to perform ankle fracture reduction surgery. 
![image of the environment.](https://github.com/CatherineMAbraham/Frac-Surg-Gym/blob/main/img/simwithcube.png)
# Installation
## From Source 
```
git clone https://github.com/CatherineMAbraham/Frac-Surg-Gym.git
cd Frac-Surg-Gym
```
## Create Conda Environment
```
conda create -n fracsurg python=3.10 -y
```
## Install Dependencies and Environment 
```
pip install -r requirements.txt
pip install -e .
```
# Usage
The environment can be loaded using Gymnasium's `gym.make()` function. This shows a basic usage with random actions, this can be replaced with any RL algorithm that utlisies the Open AI Gym Structure. 
```
import gymnasium as gym

env = gym.make('gym_fracture:fracsurg-v0')

observation,_ = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # Take a random action in the action space 
    observation, reward, terminated, truncated, info = env.step(action)

    if done or truncated:
        observation, _ = env.reset()

env.close()
```
## Agent Training 
To recreate the agent tested in the paper, you can run the following from the command line, changing the arguments to test different paramaters. 
```
 python td3.py --threshold_pos 0.005 --threshold_ori 0.05 --action_type fouractions --wandb-logging True --tensorboard-logging True --render_mode None --verbose 0
```
# Environment Information
## Action Space 
The environment has several options for action space size for researchers to use. `action_type` can be passed to the environment with one of the following flags for testing:
- `pos_only` : The agent only has a positional goal to achieve.
- `ori_only` : The agent only has a rotational goal to achieve.
- `fouractions` : The agent has a positional and rotational goal to achieve, x,y in translational, x,z in rotational.
- `sixactions` : The agent has a positional and rotational goal to achieve in all six degrees of freedom.

The agent recieves an action in every call to `step()` which represents an n dimensional vector where each element is a delta for each axes represented in the goal space. 

## Observation Space
The environment has two options, `dict` and `flat`. The `dict` option gives the observation in a dictionary format, made up of `{'observation':[], 'achieved_goal':[],'desired_goal':[]}`. To use methods such as _Hindsight Experience Replay_ this option must be enabled. The `flat` option has the same information in the _observation_ key. The observation space differs based on what `action_type` is selected and is summarised below. 

| Num | Observation |
| --- | --- |
| 0-2 | Position of the end-effector (x,y,z)|
| 3-6 | Orientation of the end-effector (x,y,z,w). |
| 7-9 | Velocity of the end-effector (x,y,z)|
| 10-18 | Pose of each joint in Panda robot. |
| 19-27 | Velocity of each joint in Panda robot. |
| 28 ^ | Euclidean distance difference of the end-effector to the goal position.|
| 29 ^^| Angle difference of the end-effector to the goal rotation.|
| 30 | Boolean describing if the pin is still held by the robot.|

^ is not included in the `ori_only` option. 
^^ is not included in the `pos_only` option.




















