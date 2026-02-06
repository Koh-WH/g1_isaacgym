import torch
from legged_gym.envs.g1.g1_env import G1Robot
from isaacgym.torch_utils import quat_rotate_inverse

class G1StandingEnv(G1Robot):
    """Environment for training G1 to stand still and maintain balance"""
    
    def __init__(self, cfg, sim_params, physics_engine, sim_device, headless):
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)
        # Store default joint positions for reward calculation
        self.standing_default_dof_pos = self.default_dof_pos.clone()

    def _resample_commands(self, env_ids):
        """Override to disable command resampling for standing task"""
        # This is automatically called by parent's reset_idx
        if len(env_ids) > 0 and self.cfg.commands.num_commands > 0:
            self.commands[env_ids] = 0.0
    
    # ============ CUSTOM REWARD FUNCTIONS ============
    
    def _reward_base_height(self):
        """Reward for maintaining target standing height with Gaussian"""
        height_error = torch.abs(self.root_states[:, 2] - self.cfg.rewards.target_height)
        return torch.exp(-(height_error / self.cfg.rewards.height_tolerance) ** 2)
    
    def _reward_orientation_upright(self):
        """Reward for maintaining upright orientation"""
        roll_pitch_error = torch.sqrt(
            self.projected_gravity[:, 0]**2 + self.projected_gravity[:, 1]**2
        )
        return torch.exp(-(roll_pitch_error / self.cfg.rewards.orientation_tolerance) ** 2)
    
    def _reward_base_lin_vel_xy(self):
        lin_vel_xy = torch.norm(self.base_lin_vel[:, :2], dim=1)
        return torch.exp(-(lin_vel_xy / self.cfg.rewards.velocity_tolerance) ** 2)
    
    def _reward_base_ang_vel_xy(self):
        ang_vel_xy = torch.norm(self.base_ang_vel[:, :2], dim=1)
        return torch.exp(-(ang_vel_xy / self.cfg.rewards.velocity_tolerance) ** 2)
    
    def _reward_base_ang_vel_z(self):
        ang_vel_z = torch.abs(self.base_ang_vel[:, 2])
        return torch.exp(-(ang_vel_z / self.cfg.rewards.velocity_tolerance) ** 2)
    
    def _reward_default_joint_pos(self):
        joint_error = torch.norm(self.dof_pos - self.standing_default_dof_pos, dim=1)
        return torch.exp(-(joint_error / self.cfg.rewards.joint_pos_tolerance) ** 2)
    
    def _reward_feet_contact(self):
        feet_in_contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        return feet_in_contact.mean(dim=1)
    
    def _reward_alive(self):
        return torch.ones(self.num_envs, device=self.device, dtype=torch.float)
    
    def _reward_feet_spacing(self):
        """
        Reward for maintaining specific Y-distance (width) between feet.
        """
        # 1. Get positions relative to base
        # rigid_body_states_view is created in _init_foot() with shape (num_envs, num_bodies, 13)
        left_foot_pos = self.rigid_body_states_view[:, self.feet_indices[0], :3] - self.root_states[:, :3]
        right_foot_pos = self.rigid_body_states_view[:, self.feet_indices[1], :3] - self.root_states[:, :3]
        
        # 2. Rotate into body frame (Critical: Handles robot turning)
        # We need this because global Y distance changes if robot rotates 90 deg.
        left_foot_body = quat_rotate_inverse(self.base_quat, left_foot_pos)
        right_foot_body = quat_rotate_inverse(self.base_quat, right_foot_pos)
        
        # 3. Calculate lateral width (Y-axis difference)
        feet_width = torch.abs(left_foot_body[:, 1] - right_foot_body[:, 1])
        
        # 4. Reward matching target width (Gaussian)
        width_error = torch.abs(feet_width - self.cfg.rewards.target_feet_width)
        return torch.exp(-(width_error / 0.05) ** 2)

    def _reward_feet_stagger(self):
        """
        Reward for keeping feet side-by-side (preventing one foot forward/back).
        Based on G1 XML: Left Hip Pitch is index 0, Right Hip Pitch is index 6.
        """
        # Hardcoded indices from your XML for speed
        left_pitch = self.dof_pos[:, 0]
        right_pitch = self.dof_pos[:, 6]
        
        # Calculate error: Ideally left_pitch == right_pitch
        stagger_error = torch.abs(left_pitch - right_pitch)
        
        # Return Gaussian Reward (1.0 = Perfect, 0.0 = Bad)
        return torch.exp(-(stagger_error / 0.1) ** 2)

    def _reward_feet_parallel(self):
        """
        Reward for keeping feet parallel (toes pointing forward).
        Based on G1 XML: Left Hip Yaw is index 2, Right Hip Yaw is index 8.
        """
        # Hardcoded indices from your XML for speed
        left_yaw = self.dof_pos[:, 2]
        right_yaw = self.dof_pos[:, 8]
        
        # We want both Yaws to be 0.0 (pointing forward)
        # Using norm to penalize any deviation from 0
        yaw_error = torch.sqrt(left_yaw**2 + right_yaw**2)
        
        return torch.exp(-(yaw_error / 0.15) ** 2)

    def _reward_feet_flat(self):
        """
        Reward for keeping feet parallel to the floor (flat).
        """
        # 1. Get Foot Quaternions: Shape [num_envs, num_feet, 4]
        feet_quat = self.rigid_body_states_view[:, self.feet_indices, 3:7]
        
        # 2. Get Gravity Vector repeated for each foot: Shape [num_envs, num_feet, 3]
        grav_vec = self.gravity_vec.unsqueeze(1).repeat(1, len(self.feet_indices), 1)
        
        # 3. FLATTEN dimensions (The Fix)
        # Combine (Env, Foot) into a single batch dim: [num_envs * num_feet, 4]
        feet_quat_flat = feet_quat.view(-1, 4)
        grav_vec_flat = grav_vec.view(-1, 3)
        
        # 4. Rotate gravity into foot frame (Now works because inputs are 2D)
        # If foot is flat, result should be [0, 0, -1]
        foot_projected_grav_flat = quat_rotate_inverse(feet_quat_flat, grav_vec_flat)
        
        # 5. Un-flatten back to [num_envs, num_feet, 3]
        foot_projected_grav = foot_projected_grav_flat.view(self.num_envs, -1, 3)
        
        # 6. Calculate deviation (X and Y components should be 0 if perfectly upright)
        deviation = torch.sum(torch.square(foot_projected_grav[:, :, :2]), dim=2)
        
        # 7. Sum error across all feet and apply Gaussian reward
        # Scale 0.1 allows small tilt, strict penalty for large tilt
        return torch.exp(-(torch.sum(deviation, dim=1) / 0.1))

    # ============ PENALTY FUNCTIONS ============
    
    def _reward_torques(self):
        return -torch.sum(torch.abs(self.torques), dim=1)
    
    def _reward_feet_air_time(self):
        # Only useful if you change 'feet_air_time' scale in config to non-zero
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        self.feet_air_time += self.dt
        air_time_penalty = torch.sum(self.feet_air_time * ~contact_filt, dim=1)
        self.feet_air_time *= ~contact_filt
        return air_time_penalty

    def _reward_orientation_yaw(self):
        """ Stronger penalty for turning """
        forward_in_body = quat_rotate_inverse(self.base_quat, self.forward_vec)
        heading_error = torch.abs(forward_in_body[:, 1])
        
        # CHANGE: Use tolerance 0.1 (very strict, approx 5 degrees)
        return torch.exp(-(heading_error / 0.1) ** 2)
    
    def _reward_stand_still(self):
        """ Reward for staying near (0,0). Wider tolerance to 'pull' robot back. """
        # Distance from center
        dist = torch.norm(self.root_states[:, :2], dim=1)
        
        # INCREASE TOLERANCE: Was 0.5, change to 1.0
        # If tolerance is too small, a robot at 0.6m gets ~0 reward and gives up.
        # At 1.0m, it still gets a gradient to pull it back.
        return torch.exp(-(dist / 1.0) ** 2)