"""
G1 Squat Environment
Save as: legged_gym/envs/g1/g1_squat_env.py
"""

import torch
import numpy as np
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1.g1_squat_config import G1SquatCfg, G1SquatCfgPPO

class G1SquatEnv(G1Robot):
    """Environment for training G1 to perform squats"""
    
    cfg: G1SquatCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Initialize squat-specific tracking
        self.last_base_height = None
    
    def _resample_commands(self, env_ids):
        """Override to disable command resampling for squat task"""
        if self.cfg.commands.num_commands > 0:
            self.commands[env_ids, :] = 0.0
    
    def reset_idx(self, env_ids):
        """Override reset to initialize squat tracking"""
        super().reset_idx(env_ids)
        if self.last_base_height is None:
            self.last_base_height = torch.zeros(self.num_envs, device=self.device)
        self.last_base_height[env_ids] = self.root_states[env_ids, 2]
    
    # ============ CUSTOM REWARD FUNCTIONS ============
    
    def _reward_squat_tracking(self):
        """
        Reward for following the squat height trajectory
        Uses a sinusoidal pattern: stand -> squat -> stand
        """
        # Calculate time in current episode
        time = self.episode_length_buf.float() * self.dt
        
        # Sinusoidal trajectory
        # -cos starts at 1 (standing), goes to -1 (squatting), back to 1 (standing)
        phase = -torch.cos(2 * np.pi * self.cfg.rewards.squat_frequency * time)
        
        # Map phase [-1, 1] to target height [squat_height, stand_height]
        # phase = -1 (bottom) -> squat_height
        # phase = +1 (top) -> stand_height
        target_height = (
            self.cfg.rewards.squat_height + 
            (self.cfg.rewards.stand_height - self.cfg.rewards.squat_height) * 
            (phase + 1.0) / 2.0
        )
        
        # Calculate height error
        current_height = self.root_states[:, 2]
        height_error = torch.abs(current_height - target_height)
        
        # Exponential reward (closer to target = higher reward)
        reward = torch.exp(-height_error**2 / self.cfg.rewards.height_tolerance**2)
        
        return reward
    
    def _reward_balance(self):
        """
        Reward for maintaining balance during squats
        Penalizes tilting (roll/pitch) and lateral drift
        """
        # Extract roll and pitch from quaternion
        quat = self.root_states[:, 3:7]
        
        roll = torch.atan2(
            2 * (quat[:, 3] * quat[:, 0] + quat[:, 1] * quat[:, 2]),
            1 - 2 * (quat[:, 0]**2 + quat[:, 1]**2)
        )
        pitch = torch.asin(
            torch.clamp(2 * (quat[:, 3] * quat[:, 1] - quat[:, 2] * quat[:, 0]), -1, 1)
        )
        
        # Penalize tilting
        orientation_error = torch.sqrt(roll**2 + pitch**2)
        orientation_reward = torch.exp(-orientation_error**2 / self.cfg.rewards.balance_tolerance**2)
        
        # Penalize lateral velocity (should only move vertically)
        lateral_vel = torch.sqrt(self.base_lin_vel[:, 0]**2 + self.base_lin_vel[:, 1]**2)
        lateral_reward = torch.exp(-lateral_vel**2 / 0.1**2)
        
        # Combined balance reward
        return orientation_reward * lateral_reward
    
    def _reward_smooth_motion(self):
        """
        Reward for smooth vertical motion
        Penalizes sudden height changes
        """
        current_height = self.root_states[:, 2]
        
        if self.last_base_height is None:
            self.last_base_height = current_height.clone()
            return torch.ones(self.num_envs, device=self.device)
        
        # Calculate height change
        height_change = torch.abs(current_height - self.last_base_height)
        
        # Expected maximum change per timestep for smooth motion
        # At 0.5 Hz, going from 0.78 to 0.45 (0.33m) in ~1 second
        # With dt=0.02s (50Hz control), max change per step ≈ 0.33/(50*1) ≈ 0.0066m
        max_smooth_change = 0.015  # Allow some margin
        
        # Penalize changes that are too rapid
        reward = torch.exp(-height_change / max_smooth_change)
        
        # Update tracking
        self.last_base_height = current_height.clone()
        
        return reward
    
    def _reward_feet_contact_squat(self):
        """
        Reward for keeping both feet in contact with ground during squats
        """
        # Check if feet have contact force
        feet_contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        
        # Reward proportional to number of feet in contact
        contact_ratio = feet_contact.float().mean(dim=1)
        
        return contact_ratio
    
    def _reward_lateral_movement(self):
        """
        Penalize horizontal (XY) movement - should only move vertically
        """
        lateral_vel = torch.sqrt(self.base_lin_vel[:, 0]**2 + self.base_lin_vel[:, 1]**2)
        return lateral_vel**2
    
    def post_physics_step(self):
        """Override to update squat-specific state tracking"""
        super().post_physics_step()
        
        # Additional logging for debugging (optional)
        if self.common_step_counter % 100 == 0:
            time = self.episode_length_buf[0].float() * self.dt
            phase = -torch.cos(2 * np.pi * self.cfg.rewards.squat_frequency * time)
            target_height = (
                self.cfg.rewards.squat_height + 
                (self.cfg.rewards.stand_height - self.cfg.rewards.squat_height) * 
                (phase + 1.0) / 2.0
            )
            current_height = self.root_states[0, 2].item()
            # Uncomment for debugging:
            # print(f"Time: {time:.2f}s, Target: {target_height:.3f}m, Current: {current_height:.3f}m")
    
    # Inherited reward functions from G1Robot and LeggedRobot
    # will be scaled according to the config in g1_squat_config.py