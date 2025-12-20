"""
G1 Standing Environment
Save as: legged_gym/envs/g1/g1_standing_env.py
"""

import torch
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1.g1_standing_config import G1StandingCfg, G1StandingCfgPPO

class G1StandingEnv(G1Robot):
    """Environment for training G1 to stand still and maintain balance"""
    
    cfg: G1StandingCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
    
    def _resample_commands(self, env_ids):
        """Override to disable command resampling for standing task"""
        # Standing task has no commands, so we just set them to zero
        if self.cfg.commands.num_commands > 0:
            self.commands[env_ids, :] = 0.0
        # If num_commands is 0, commands tensor might not exist or be empty, so we skip
    
    # ============ CUSTOM REWARD FUNCTIONS ============
    
    def _reward_base_height(self):
        """Reward for maintaining target standing height"""
        height_error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.target_height)
        reward = torch.exp(-height_error**2 / self.cfg.rewards.height_sigma**2)
        return reward
    
    def _reward_orientation(self):
        """Reward for maintaining upright orientation"""
        # Calculate roll and pitch from quaternion
        # Quaternion: [x, y, z, w] format in root_states[:, 3:7]
        quat = self.root_states[:, 3:7]
        
        # Convert to roll and pitch
        roll = torch.atan2(
            2 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 0]**2 + quat[:, 1]**2)
        )
        pitch = torch.asin(
            torch.clamp(2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1, 1)
        )
        
        # Penalize deviation from upright (roll=0, pitch=0)
        orientation_error = torch.sqrt(roll**2 + pitch**2)
        reward = torch.exp(-orientation_error**2 / self.cfg.rewards.orientation_sigma**2)
        return reward
    
    def _reward_base_stability(self):
        """Reward for minimal base XY movement"""
        # Only penalize horizontal movement, allow vertical for balance
        lin_vel_xy = torch.sqrt(self.base_lin_vel[:, 0]**2 + self.base_lin_vel[:, 1]**2)
        reward = torch.exp(-lin_vel_xy**2 / self.cfg.rewards.base_motion_sigma**2)
        return reward
    
    # Other reward functions are inherited from G1Robot and LeggedRobot
    # They will be scaled according to the config in g1_standing_config.py