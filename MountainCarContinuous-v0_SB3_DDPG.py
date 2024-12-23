import gymnasium as gym
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.utils import constant_fn
import numpy as np
op={
    'action_noise': OrnsteinUhlenbeckActionNoise,
    'gradient_steps': 1,
    'train_freq': 1,
    'learning_rate': 1e-3,
    'batch_size': 256,
    'policy_kwargs': dict(net_arch=[400, 300]),
}
env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")
n_actions = env.action_space.shape[0]
op["action_noise"] = OrnsteinUhlenbeckActionNoise(
                    mean=np.zeros(n_actions),
                    sigma=0.5 * np.ones(n_actions),
                )

model = DDPG(policy='MlpPolicy',env=env,verbose=1,**op)
model.learn(total_timesteps=10_0000)
