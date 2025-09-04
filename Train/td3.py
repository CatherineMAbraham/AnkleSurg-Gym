import gymnasium as gym
from stable_baselines3 import TD3, HerReplayBuffer
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import tensorboard
from stable_baselines3.common.monitor import Monitor
import wandb
import numpy as np
from typing import Callable
import datetime
import argparse

def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func


def train(threshold_pos:float=0.001, threshold_ori:float=np.deg2rad(6), 
          action_type:str='pos_only', wandb_logging:bool=True, tensorboard_logging:bool=True, render_mode:str=None, verbose:int=0):
    """Train the TD3 model with the specified parameters.
    Args:
        threshold_pos (float): The position threshold for the environment.
        threshold_ori (float): The orientation threshold for the environment.
        action_type (str): The action type for the environment.
        wandb_logging (bool): Whether to enable Weights & Biases logging.
        tensorboard_logging (bool): Whether to enable TensorBoard logging.
        render_mode (str): The render mode for the environment.
        verbose (int): The verbosity level for logging.
    """
    x = datetime.datetime.now()
    train_date = x.strftime('%m%d%H%M')
    action_type = action_type
    threshold_pos = threshold_pos
    threshold_ori = threshold_ori
    render_mode = render_mode
    verbose = verbose
    if wandb_logging:
        assert tensorboard_logging, "To use wandb-logging, you must set tensorboard-logging to True and be logged into wandb."
        wandb.init(project="Test", name = (f'{train_date}-{action_type}-{threshold_pos}-{threshold_ori}'),sync_tensorboard=True, save_code=True)  # Initialize W&B

    env = gym.make('gym_fracture:anklesurg-v0', 
                   reward_type='sparse', 
                   max_steps=100, 
                   horizon='variable', 
                   obs_type='dict', 
                   distance_threshold_pos= threshold_pos,
                   dv =0.05,
                   distance_threshold_ori= threshold_ori,
                   action_type=action_type, render_mode=render_mode)
    
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(env.action_space.shape[0]), 
                                              sigma=0.02 * np.ones(env.action_space.shape[0]))
    
    policy_kwargs = dict(net_arch=[256, 256,256])
    if tensorboard_logging:
        model = TD3(policy="MultiInputPolicy", 
                    env=env,verbose=verbose,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(n_sampled_goal=4),
                    learning_rate=linear_schedule(0.0003),
                    train_freq=1,
                    buffer_size=1000000,
                    learning_starts=500,
                    batch_size=256,
                    tau= 0.005,
                    gamma=0.93,
                    policy_kwargs=policy_kwargs,
                    gradient_steps=-1,
                    seed=3, action_noise=action_noise,
                    tensorboard_log='./logs')
    else:
        model = TD3(policy="MultiInputPolicy", 
                    env=env,verbose=verbose,
                    replay_buffer_class=HerReplayBuffer,
                    replay_buffer_kwargs=dict(n_sampled_goal=4),
                    learning_rate=linear_schedule(0.0003),
                    train_freq=1,
                    buffer_size=1000000,
                    learning_starts=500,
                    batch_size=256,
                    tau= 0.005,
                    gamma=0.93,
                    policy_kwargs=policy_kwargs,
                    gradient_steps=-1,
                    seed=3, action_noise=action_noise)

    # Separate evaluation env
    eval_env = Monitor(env)

    eval_callback = EvalCallback(eval_env,  eval_freq=10000, 
                                deterministic=True, n_eval_episodes=20)
    
    model.learn(800000, callback=eval_callback)
    model.save(f'./model-{train_date}-{action_type}-{threshold_pos}-{threshold_ori}')

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train TD3 model with specified thresholds and action type.')
    parser.add_argument('--threshold_pos', type=float, default=0.005, help='Position threshold for the environment.')
    parser.add_argument('--threshold_ori', type=float, default=0.05, help='Orientation threshold for the environment.')
    parser.add_argument('--action_type', type=str, default='fouractions', help='Type of action to use in the environment.')
    parser.add_argument('--wandb-logging', type=bool, default=False, help='Enable or disable Weights & Biases logging. Must set tensorboard-logging to True and logged into wandb.')
    parser.add_argument('--tensorboard-logging', type=bool, default=True, help='Enable or disable TensorBoard logging.')
    parser.add_argument('--render_mode', type=str, default=None, help='Render mode for the environment.')
    parser.add_argument('--verbose', type=int, default=0, help='Verbosity level for logging.')
    args = parser.parse_args()

    train(threshold_pos=args.threshold_pos, 
          threshold_ori=args.threshold_ori, 
          action_type=args.action_type, 
          wandb_logging=args.wandb_logging, 
          tensorboard_logging=args.tensorboard_logging, 
          render_mode=args.render_mode, 
          verbose=args.verbose)
