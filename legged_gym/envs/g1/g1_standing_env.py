"""
G1 Standing Environment - Improved Version
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
        
        # Store default joint positions for reward calculation
        # After super().__init__(), self.default_dof_pos is already set by base class
        # We'll save it for our custom reward
        self.standing_default_dof_pos = self.default_dof_pos.clone()
    
    def _resample_commands(self, env_ids):
        """Override to disable command resampling for standing task"""
        if len(env_ids) > 0 and self.cfg.commands.num_commands > 0:
            self.commands[env_ids] = 0.0
    
    def reset_idx(self, env_ids):
        """Reset environments and ensure commands are zero"""
        super().reset_idx(env_ids)
        if len(env_ids) > 0 and self.cfg.commands.num_commands > 0:
            self.commands[env_ids] = 0.0
    
    # ============ CUSTOM REWARD FUNCTIONS ============
    
    def _reward_base_height(self):
        """Reward for maintaining target standing height with Gaussian"""
        height_error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.target_height)
        return torch.exp(-(height_error / self.cfg.rewards.height_tolerance) ** 2)
    
    def _reward_orientation_upright(self):
        """Reward for maintaining upright orientation (roll and pitch near zero)"""
        # Use projected_gravity: [0, 0, -1] when upright
        # Roll and pitch deviations captured by x and y components
        roll_pitch_error = torch.sqrt(
            self.projected_gravity[:, 0]**2 + self.projected_gravity[:, 1]**2
        )
        return torch.exp(-(roll_pitch_error / self.cfg.rewards.orientation_tolerance) ** 2)
    
    def _reward_base_lin_vel_xy(self):
        """Reward for minimal horizontal base velocity"""
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return torch.exp(-(lin_vel_xy / self.cfg.rewards.velocity_tolerance) ** 2)
    
    def _reward_base_ang_vel_xy(self):
        """Reward for minimal roll/pitch angular velocity"""
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        return torch.exp(-(ang_vel_xy / self.cfg.rewards.velocity_tolerance) ** 2)
    
    def _reward_base_ang_vel_z(self):
        """Reward for minimal yaw angular velocity"""
        ang_vel_z = torch.abs(self.base_ang_vel[:, 2])
        return torch.exp(-(ang_vel_z / self.cfg.rewards.velocity_tolerance) ** 2)
    
    def _reward_default_joint_pos(self):
        """Reward for keeping joints near default standing configuration"""
        joint_error = torch.norm(self.dof_pos - self.standing_default_dof_pos, dim=1)
        return torch.exp(-(joint_error / self.cfg.rewards.joint_pos_tolerance) ** 2)
    
    def _reward_feet_contact(self):
        """Reward for keeping both feet in contact with ground"""
        # Assuming feet are the last two bodies in contact sensor
        # Adjust indices based on your robot's contact sensor setup
        # This is a simplified version - adjust based on your contact sensor setup
        contact_forces = self.contact_forces[:, self.feet_indices, 2]  # Z-forces
        
        # Reward when feet have contact (force > threshold)
        feet_in_contact = (contact_forces > 1.0).float()
        # Return average contact across both feet
        return feet_in_contact.mean(dim=1)
    
    def _reward_alive(self):
        """Bonus for staying alive (not terminating)"""
        return torch.ones(self.num_envs, device=self.device, dtype=torch.float)
    
    # ============ PENALTY FUNCTIONS ============
    
    def _reward_torques(self):
        """Penalize large torques (negative reward)"""
        return -torch.sum(torch.abs(self.torques), dim=1)
    
    def _reward_feet_air_time(self):
        """Penalize feet being in the air (negative reward)"""
        # Get contact state for feet
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        # Calculate air time (time since last contact)
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        # Penalize any air time
        air_time_penalty = torch.sum(self.feet_air_time * ~contact_filt, dim=1)
        self.feet_air_time *= ~contact_filt
        return air_time_penalty
    
    def _reward_base_lin_acc(self):
        """Penalize linear acceleration (jerking motion)"""
        return torch.norm(self.last_actions - self.actions, dim=1)
    
    def _reward_base_ang_acc(self):
        """Penalize angular acceleration (rotational jerking)"""
        if not hasattr(self, 'last_base_ang_vel'):
            self.last_base_ang_vel = self.base_ang_vel.clone()
            return torch.zeros(self.num_envs, device=self.device)
        
        ang_acc = torch.norm(self.base_ang_vel - self.last_base_ang_vel, dim=1) / self.dt
        self.last_base_ang_vel = self.base_ang_vel.clone()
        return ang_acc
    
    # Other reward functions (lin_vel_z, ang_vel_xy, dof_vel, dof_acc, 
    # action_rate, dof_pos_limits, collision, hip_pos) are inherited 
    # from G1Robot and LeggedRobot base classes