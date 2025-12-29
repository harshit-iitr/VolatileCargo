import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# --- 1. THE GYM ENVIRONMENT ---
class QuarterCarEnv(gym.Env):
    def __init__(self, road_data_dict):
        super(QuarterCarEnv, self).__init__()
        
        # Physics Constants
        self.ms = 290.0
        self.mu = 59.0
        self.ks = 16000.0
        self.kt = 190000.0
        self.c_min = 800.0
        self.c_max = 3500.0
        self.dt = 0.005
        
        # Data
        self.road_data_dict = road_data_dict
        self.profile_names = list(road_data_dict.keys())
        self.current_profile_data = None
        self.time_step = 0
        
        # Action Space: Normalized Damping [-1, 1] mapped to [c_min, c_max]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        
        # Observation Space: 
        # [z_s, v_s, z_u, v_u, acc_s, acc_u, 
        #  buffer_0, buffer_1, buffer_2, buffer_3 (The 4 delay steps)]
        # We include the delay buffer so the agent knows "what's coming" in the actuator pipe
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Pick a random profile to train on (or cycle through them)
        p_name = np.random.choice(self.profile_names)
        self.current_profile_data = self.road_data_dict[p_name]
        
        # Reset State
        self.state = np.zeros(4) # [zs, vs, zu, vu]
        self.time_step = 0
        self.delay_buffer = [self.c_min] * 4 # Reset buffer
        self.prev_acc_s = 0.0 # For Jerk calc
        
        return self._get_obs(), {}

    def _get_obs(self):
        # Scale inputs slightly to help Neural Net (normalize roughly to [-1, 1])
        # This isn't strictly necessary but helps convergence
        zs, vs, zu, vu = self.state
        
        # We calculate accelerations for the observation
        # Note: We need the LAST applied c, which is at buffer index 0
        c_applied = self.delay_buffer[0]
        r_t = self.current_profile_data[self.time_step] if self.time_step < len(self.current_profile_data) else 0
        
        f_susp = self.ks * (zu - zs) + c_applied * (vu - vs)
        f_tire = self.kt * (r_t - zu)
        acc_s = f_susp / self.ms
        acc_u = (-f_susp + f_tire) / self.mu
        
        # Normalize buffer for observation
        norm_buffer = [(c - 2150)/1350 for c in self.delay_buffer]
        
        obs = np.array([zs, vs, zu, vu, acc_s, acc_u] + norm_buffer, dtype=np.float32)
        return obs

    def step(self, action):
        # 1. Convert Action [-1, 1] -> [800, 3500]
        c_cmd = 2150.0 + (action[0] * 1350.0)
        c_cmd = np.clip(c_cmd, self.c_min, self.c_max)
        
        # 2. Handle Delay
        self.delay_buffer.append(c_cmd)
        c_applied = self.delay_buffer.pop(0)
        
        # 3. Physics Step
        r_t = self.current_profile_data[self.time_step]
        zs, vs, zu, vu = self.state
        
        f_susp = self.ks * (zu - zs) + c_applied * (vu - vs)
        f_tire = self.kt * (r_t - zu)
        
        acc_s = f_susp / self.ms
        acc_u = (-f_susp + f_tire) / self.mu
        
        # Integration
        vs_new = vs + acc_s * self.dt
        vu_new = vu + acc_u * self.dt
        zs_new = zs + vs_new * self.dt
        zu_new = zu + vu_new * self.dt
        
        self.state = np.array([zs_new, vs_new, zu_new, vu_new])
        
        # 4. Calculate Reward (Negative Cost)
        # We want to minimize Displacement and Jerk.
        # Jerk approx
        jerk = (acc_s - self.prev_acc_s) / self.dt
        self.prev_acc_s = acc_s
        
        # Reward Function: The Secret Sauce
        # We weigh Jerk heavily because that's where you are failing
        reward = - (1.0 * (zs**2) + 0.5 * (jerk**2) + 0.05 * (abs(jerk)**3))
        
        # 5. Check Done
        self.time_step += 1
        done = self.time_step >= len(self.current_profile_data) - 1
        truncated = False
        
        return self._get_obs(), reward, done, truncated, {}
