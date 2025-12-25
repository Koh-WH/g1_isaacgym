from legged_gym.envs.base.legged_robot import LeggedRobot
# FIXED: Pointing to the specific G1 env location
from legged_gym.envs.g1.g1_env import G1Robot
from isaacgym.torch_utils import *
import torch
import numpy as np

# RENAMED: Class name now matches your __init__.py 'G1SquattingEnv'
class G1SquattingEnv(G1Robot):
    
    def _post_physics_step_callback(self):
        # Run parent logic first (updates feet state, checks termination)
        super()._post_physics_step_callback()

        # --- Overwrite Phase logic for Squatting ---
        # Slower period for squatting stability (e.g., 2.0s period)
        period = 1.0 / self.cfg.rewards.squat_freq
        self.phase = (self.episode_length_buf * self.dt) % period / period
        
        # Calculate Target Height (Sine Wave)
        # h = h_mean + Amp * sin(2*pi*phase)
        mean_h = self.cfg.rewards.base_height_target
        amp = self.cfg.rewards.squat_amp
        self.target_height = mean_h + amp * torch.sin(2 * torch.pi * self.phase)

        # Synchronize legs (squatting involves both legs moving together)
        self.phase_left = self.phase
        self.phase_right = self.phase
        self.leg_phase = torch.cat([self.phase_left.unsqueeze(1), self.phase_right.unsqueeze(1)], dim=-1)

    def _reward_track_squat_z(self):
        # Reward tracking the sine wave trajectory
        root_states = self.root_states
        base_z = root_states[:, 2]
        error = torch.square(base_z - self.target_height)
        return torch.exp(-error / 0.05) # Gaussian kernel

    def _reward_feet_stuck(self):
        # Reward if BOTH feet are in contact with the ground
        contact = torch.norm(self.contact_forces[:, self.feet_indices, :3], dim=2) > 1.
        # Check if both feet (indices 0 and 1) are in contact
        both_feet_contact = (contact[:, 0] & contact[:, 1])
        return both_feet_contact.float()

    def _reward_feet_slip(self):
        # Penalize any velocity of the feet (they should be planted)
        foot_vel = torch.norm(self.feet_vel, dim=2)
        return torch.sum(torch.square(foot_vel), dim=1)

    def _reward_tracking_lin_vel(self):
        # Override to ensure it returns 0 (disable walking logic)
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_feet_air_time(self):
        # Override to 0, squatting has 0 air time
        return torch.zeros(self.num_envs, device=self.device)