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
## Usage
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
