"""
G1 Squatting Environment
Save as: legged_gym/envs/g1/g1_squatting_env.py
"""

import torch
import numpy as np
from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1.g1_squatting_config import G1SquattingCfg, G1SquattingCfgPPO

class G1SquattingEnv(G1Robot):
    """Environment for training G1 to perform continuous squats"""
    
    cfg: G1SquattingCfg
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        # Initialize squat phase tracker
        self.squat_phase = None
        self.last_height = None
        
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        
        # Store standing default positions
        self.standing_default_dof_pos = self.default_dof_pos.clone()
        
        # Initialize squat phase (0 to 2π, where 0 and 2π = standing, π = squatting)
        self.squat_phase = torch.zeros(self.num_envs, device=self.device, dtype=torch.float)
        self.last_height = self.root_states[:, 2].clone()
    
    def reset_idx(self, env_ids):
        """Reset environments"""
        super().reset_idx(env_ids)
        if len(env_ids) > 0:
            # Zero out commands
            if self.cfg.commands.num_commands > 0:
                self.commands[env_ids] = 0.0
            # Reset squat phase to random position in cycle
            self.squat_phase[env_ids] = torch.rand(len(env_ids), device=self.device) * 2 * np.pi
            self.last_height[env_ids] = self.root_states[env_ids, 2]
    
    def _resample_commands(self, env_ids):
        """Override to disable command resampling"""
        if len(env_ids) > 0 and self.cfg.commands.num_commands > 0:
            self.commands[env_ids] = 0.0
    
    def post_physics_step(self):
        """Update squat phase and call parent"""
        # Update squat phase (continuous cycling)
        phase_increment = 2 * np.pi * self.cfg.rewards.squat_frequency * self.dt
        self.squat_phase += phase_increment
        self.squat_phase = torch.fmod(self.squat_phase, 2 * np.pi)
        
        super().post_physics_step()
    
    def compute_observations(self):
        """Add squat phase to observations"""
        super().compute_observations()
        
        # Add squat phase as sine and cosine for smooth cycling
        squat_phase_obs = torch.stack([
            torch.sin(self.squat_phase),
            torch.cos(self.squat_phase)
        ], dim=1)
        
        # Note: This assumes obs_buf has room for 2 extra values
        # If base class obs is 47, we need to ensure num_observations = 48
        # We'll append the phase observation
        if self.obs_buf.shape[1] == self.cfg.env.num_observations - 1:
            # Append squat phase (using just sine for simplicity)
            self.obs_buf = torch.cat([
                self.obs_buf, 
                torch.sin(self.squat_phase).unsqueeze(1)
            ], dim=1)
    
    def _get_target_height(self):
        """Calculate target height based on squat phase"""
        # Sine wave: standing (0) -> squat down (π) -> standing (2π)
        # Height varies from standing_height to squat_height
        normalized_height = (1 + torch.cos(self.squat_phase)) / 2  # 1 at 0, 0 at π, 1 at 2π
        target_height = (self.cfg.rewards.squat_height + 
                        (self.cfg.rewards.standing_height - self.cfg.rewards.squat_height) * normalized_height)
        return target_height
    
    # ============ CUSTOM REWARD FUNCTIONS ============
    
    def _reward_track_squat_height(self):
        """Reward for tracking the target squat height trajectory"""
        target_height = self._get_target_height()
        current_height = self.root_states[:, 2]
        height_error = torch.abs(current_height - target_height)
        return torch.exp(-(height_error / self.cfg.rewards.height_tolerance) ** 2)
    
    def _reward_orientation_upright(self):
        """Reward for maintaining upright orientation during squats"""
        roll_pitch_error = torch.sqrt(
            self.projected_gravity[:, 0]**2 + self.projected_gravity[:, 1]**2
        )
        return torch.exp(-(roll_pitch_error / self.cfg.rewards.orientation_tolerance) ** 2)
    
    def _reward_hip_knee_coordination(self):
        """Reward proper hip and knee bending coordination"""
        # Get hip and knee joint positions
        hip_joints = self.cfg.rewards.hip_joints
        knee_joints = self.cfg.rewards.knee_joints
        
        # During squatting, hips should flex (negative angle) and knees should flex (positive angle)
        # Calculate expected joint angles based on squat depth
        squat_depth_ratio = (1 - torch.cos(self.squat_phase)) / 2  # 0 at standing, 1 at deepest squat
        
        # Expected hip flexion (negative values)
        expected_hip_angle = -squat_depth_ratio * 0.8  # Up to ~45 degrees
        # Expected knee flexion (positive values)  
        expected_knee_angle = squat_depth_ratio * 1.2   # Up to ~70 degrees
        
        # Calculate errors
        hip_error = torch.mean(torch.abs(self.dof_pos[:, hip_joints] - expected_hip_angle), dim=1)
        knee_error = torch.mean(torch.abs(self.dof_pos[:, knee_joints] - expected_knee_angle), dim=1)
        
        total_error = hip_error + knee_error
        return torch.exp(-total_error**2 / 0.5)
    
    def _reward_smooth_height_change(self):
        """Reward smooth height transitions"""
        current_height = self.root_states[:, 2]
        height_velocity = torch.abs(current_height - self.last_height) / self.dt
        self.last_height = current_height.clone()
        
        # Expected velocity based on squat phase derivative
        target_velocity = torch.abs(
            -torch.sin(self.squat_phase) * np.pi * self.cfg.rewards.squat_frequency * 
            (self.cfg.rewards.standing_height - self.cfg.rewards.squat_height)
        )
        
        velocity_error = torch.abs(height_velocity - target_velocity)
        return torch.exp(-velocity_error**2 / 0.1)
    
    def _reward_base_lin_vel_xy(self):
        """Reward for minimal horizontal base velocity"""
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return torch.exp(-lin_vel_xy**2 / 0.1)
    
    def _reward_base_ang_vel_xy(self):
        """Reward for minimal roll/pitch angular velocity"""
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        return torch.exp(-ang_vel_xy**2 / 0.1)
    
    def _reward_base_ang_vel_z(self):
        """Reward for minimal yaw angular velocity"""
        ang_vel_z = torch.abs(self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_z**2 / 0.1)
    
    def _reward_feet_contact(self):
        """Reward for keeping both feet in contact with ground"""
        contact_forces = self.contact_forces[:, self.feet_indices, 2]
        feet_in_contact = (contact_forces > 1.0).float()
        return feet_in_contact.mean(dim=1)
    
    def _reward_alive(self):
        """Bonus for staying alive"""
        return torch.ones(self.num_envs, device=self.device, dtype=torch.float)
    
    def _reward_termination(self):
        """Penalize episode termination"""
        return self.reset_buf.float()
    
    def _reward_torques(self):
        """Penalize large torques"""
        return torch.sum(torch.abs(self.torques), dim=1)
    
    # Other reward functions inherited from base classes