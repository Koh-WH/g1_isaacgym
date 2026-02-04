from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch
import numpy as np


class G1_1Env(LeggedRobot):
    
    def _get_noise_scale_vec(self, cfg):
        """Noise scaling for 98-dim observations"""
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        
        idx = 0
        noise_vec[idx:idx+3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        idx += 3
        noise_vec[idx:idx+3] = noise_scales.gravity * noise_level
        idx += 3
        # noise_vec[idx:idx+3] = 0.
        noise_vec[idx:idx+3] = noise_scales.lin_vel * noise_level * self.obs_scales.lin_vel
        idx += 3
        noise_vec[idx:idx+29] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        idx += 29
        noise_vec[idx:idx+29] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        idx += 29
        noise_vec[idx:idx+29] = 0.
        idx += 29
        noise_vec[idx:idx+2] = 0.
        
        return noise_vec

    def _init_foot(self):
        """Initialize foot tracking buffers"""
        self.feet_num = len(self.feet_indices)
        
        rigid_body_state = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_state)
        self.rigid_body_states_view = self.rigid_body_states.view(self.num_envs, -1, 13)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _init_arm_and_waist(self):
        """Initialize arm and waist tracking buffers"""
        self.waist_dof_indices = torch.tensor([12, 13, 14], device=self.device)
        self.arm_dof_indices = torch.arange(15, 29, device=self.device)
        self.left_arm_dof_indices = torch.arange(15, 22, device=self.device)
        self.right_arm_dof_indices = torch.arange(22, 29, device=self.device)
        
        # For torque tracking (needed for some rewards)
        self.last_torques = torch.zeros_like(self.torques)
        
    def _init_buffers(self):
        """Initialize all buffers"""
        super()._init_buffers()
        self._init_foot()
        self._init_arm_and_waist()
        
        self.standing_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        
        print(f"\n{'='*60}")
        print(f"G1 29DOF Stage 1: BASIC STANDING - CORRECTED VERSION")
        print(f"{'='*60}")
        print(f"Observation dims: {self.obs_buf.shape[-1]} (expected: {self.cfg.env.num_observations})")
        print(f"Privileged obs dims: {self.privileged_obs_buf.shape[-1] if self.privileged_obs_buf is not None else 'None'} (expected: {self.cfg.env.num_privileged_obs})")
        print(f"Action dims: {self.actions.shape[-1]} (expected: {self.cfg.env.num_actions})")
        print(f"Active rewards: ")
        print(f"  - Core: alive, termination")
        print(f"  - Stability: orientation, base_height, both_feet_contact, single_foot_penalty")
        print(f"  - Contact: contact_no_vel, feet_stumble")
        print(f"  - Movement: ang_vel_xy, lin_vel_z")
        print(f"  - Control: dof_vel, dof_acc, action_rate, torques")
        print(f"  - Safety: collision, dof_pos_limits, torque_limits")
        print(f"  - Posture: dof_pos_default, feet_parallel, ankle_stability")
        print(f"{'='*60}\n")

    def update_feet_state(self):
        """Update foot state from rigid body states"""
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.feet_state = self.rigid_body_states_view[:, self.feet_indices, :]
        self.feet_pos = self.feet_state[:, :, :3]
        self.feet_vel = self.feet_state[:, :, 7:10]
        
    def _post_physics_step_callback(self):
        """Post-physics callback"""
        self.update_feet_state()
        
        # Track last torques for rewards
        self.last_torques[:] = self.torques[:]
        
        # Phase tracking (simple, not used in Stage 1)
        period = 1.0
        self.phase = (self.episode_length_buf * self.dt) % period / period
        
        # Track standing quality
        upright_mask = (torch.abs(self.rpy[:, 0]) < 0.15) & (torch.abs(self.rpy[:, 1]) < 0.15)
        self.standing_time += self.dt * upright_mask.float()
        self.standing_time *= ~self.reset_buf
        
        return super()._post_physics_step_callback()
    
    def check_termination(self):
        self.reset_buf = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        # Hard fall only
        self.reset_buf |= self.base_pos[:, 2] < 0.25

        # Extreme tilt only
        self.reset_buf |= torch.abs(self.rpy[:, 0]) > 1.8
        self.reset_buf |= torch.abs(self.rpy[:, 1]) > 1.8

        # Timeout
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf

    
    def reset_idx(self, env_ids):
        """Reset environments"""
        if len(env_ids) == 0:
            return
        
        # Log successes
        timeout_ids = env_ids[self.time_out_buf[env_ids]]
        if len(timeout_ids) > 0:
            avg_upright = self.standing_time[timeout_ids].mean()
            avg_pitch = torch.abs(self.rpy[timeout_ids, 1]).mean() * 180 / 3.14159
            avg_roll = torch.abs(self.rpy[timeout_ids, 0]).mean() * 180 / 3.14159
            # print(f"✅ Stage 1 SUCCESS! {len(timeout_ids)} robots stood for 30s! "
                #   f"Upright: {avg_upright:.1f}s, Avg tilt: {avg_pitch:.1f}°pitch/{avg_roll:.1f}°roll")
        
        # Reset tracking
        self.standing_time[env_ids] = 0.
        
        # Standard reset
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        
        self.actions[env_ids] = 0.
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]["rew_" + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf
    
    def _resample_commands(self, env_ids):
        """Commands always zero for standing"""
        self.commands[env_ids, :] = 0.
    
    def compute_observations(self):
        """Compute 98-dim observations"""
        sin_phase = torch.sin(2 * np.pi * self.phase).unsqueeze(1)
        cos_phase = torch.cos(2 * np.pi * self.phase).unsqueeze(1)
        
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase,
        ), dim=-1)
        
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            sin_phase,
            cos_phase,
        ), dim=-1)
        
        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    # ========================================
    # REWARD FUNCTIONS - CORRECTED & IMPROVED
    # ========================================

    def _reward_alive(self):
        """Reward for staying alive"""
        return torch.ones(self.num_envs, dtype=torch.float, device=self.device)
    
    def _reward_termination(self):
        """Penalty for falling/terminating"""
        return self.reset_buf.float()
    
    def _reward_both_feet_contact(self):
        """Reward for having both feet in contact with ground"""
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        both_contact = left_contact & right_contact
        return both_contact.float()
    
    def _reward_single_foot_penalty(self):
        """Penalty for single foot contact (standing on one leg)"""
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        single_foot = torch.logical_xor(left_contact, right_contact)
        return single_foot.float()
    
    def _reward_contact_no_vel(self):
        """Penalty for foot sliding while in contact"""
        left_contact = self.contact_forces[:, self.feet_indices[0], 2] > 1.0
        right_contact = self.contact_forces[:, self.feet_indices[1], 2] > 1.0
        
        # Get foot velocities in XY plane
        left_foot_vel = torch.norm(self.feet_vel[:, 0, :2], dim=1)
        right_foot_vel = torch.norm(self.feet_vel[:, 1, :2], dim=1)
        
        # Penalize sliding when in contact
        left_slip = torch.where(left_contact, left_foot_vel, torch.zeros_like(left_foot_vel))
        right_slip = torch.where(right_contact, right_foot_vel, torch.zeros_like(right_foot_vel))
        
        return left_slip + right_slip
    
    def _reward_feet_stumble(self):
        """Penalize feet hitting vertical surfaces (walls, obstacles)"""
        # Check if horizontal contact forces exceed vertical forces
        return torch.any(
            torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) > \
            5 * torch.abs(self.contact_forces[:, self.feet_indices, 2]), 
            dim=1
        ).float()
    
    def _reward_orientation(self):
        """Penalty for body tilt (keep torso upright)"""
        # Projected gravity should be [0, 0, -1] when upright
        return torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_base_height(self):
        """Penalty for deviation from target standing height"""
        return torch.square(self.base_pos[:, 2] - self.cfg.rewards.base_height_target)
    
    def _reward_ang_vel_xy(self):
        """Penalty for body angular velocity (roll/pitch rates)"""
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
    
    def _reward_lin_vel_z(self):
        """Penalty for vertical bouncing"""
        return torch.square(self.base_lin_vel[:, 2])
    
    def _reward_dof_vel(self):
        """Penalty for high joint velocities"""
        return torch.sum(torch.square(self.dof_vel), dim=1)
    
    def _reward_dof_acc(self):
        """Penalty for high joint accelerations (encourage smooth motion)"""
        return torch.sum(torch.square((self.dof_vel - self.last_dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        """Penalty for rapid action changes"""
        return torch.sum(torch.square(self.actions - self.last_actions), dim=1)
    
    def _reward_torques(self):
        """Penalty for high torques (encourage energy efficiency)"""
        return torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_collision(self):
        """Penalty for body collisions"""
        return torch.sum(self.contact_forces[:, self.penalised_contact_indices, 2] > 0.1, dim=1).float()
    
    def _reward_dof_pos_limits(self):
        """Penalty for approaching joint limits"""
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.sum(out_of_limits, dim=1)
    
    def _reward_torque_limits(self):
        """Penalty for approaching torque limits (actuator saturation)"""
        max_torques = torch.abs(self.torque_limits)
        exceeded = (torch.abs(self.torques) - max_torques * self.cfg.rewards.soft_torque_limit).clip(min=0.)
        return torch.sum(exceeded, dim=1)
    
    def _reward_dof_pos_default(self):
        """Gentle penalty for deviating from default joint positions"""
        return torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_feet_parallel(self):
        """Reward for keeping feet parallel (not staggered stance)
        
        Penalizes asymmetric hip pitch angles which cause one foot forward, one back.
        """
        # Hip pitch joints: index 0 (left) and 6 (right)
        left_hip_pitch = self.dof_pos[:, 0]
        right_hip_pitch = self.dof_pos[:, 6]
        
        # They should be symmetric (same value)
        asymmetry = torch.square(left_hip_pitch - right_hip_pitch)
        
        return asymmetry
    
    def _reward_ankle_stability(self):
        """Reward for stable ankle control (reduce jitter/oscillation)
        
        Encourages smooth, controlled ankle movements rather than rapid corrections.
        Uses exponential decay so the penalty is bounded and gradual.
        """
        # Ankle pitch indices: 4 (left), 10 (right)
        ankle_indices = torch.tensor([4, 10], device=self.device)
        
        # Get ankle velocities
        ankle_vel = torch.abs(self.dof_vel[:, ankle_indices])
        
        # Exponential reward: returns ~2.0 when still, decays toward 0 as velocity increases
        # sigma=0.5 means velocities < 0.5 rad/s get good rewards
        # This is BOUNDED (max value = 2.0) unlike squared which can explode
        return torch.sum(torch.exp(-torch.square(ankle_vel / 0.5)), dim=1)
    
    def _reward_base_lin_vel_xy(self):
        """Penalty for horizontal base movement"""
        return torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
    
    def _reward_base_ang_vel_z(self):
        """Penalty for yaw rotation"""
        return torch.square(self.base_ang_vel[:, 2])
    
    def _reward_dof_vel_limits(self):
        """Velocity limit penalty"""
        return torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _reward_hip_pos(self):
        """Hip position tracking"""
        hip_default = self.default_dof_pos[:, :6]
        hip_current = self.dof_pos[:, :6]
        return torch.sum(torch.square(hip_current - hip_default), dim=1)
    
    def _reward_knee_alignment(self):
        """Knee bend alignment"""
        left_knee = self.dof_pos[:, 3]
        right_knee = self.dof_pos[:, 9]
        return torch.square(left_knee - right_knee)
    
    def _reward_waist_position(self):
        """Waist neutral position"""
        waist_pos = self.dof_pos[:, self.waist_dof_indices]
        waist_default = self.default_dof_pos[:, self.waist_dof_indices]
        return torch.sum(torch.square(waist_pos - waist_default), dim=1)
    
    def _reward_arm_position(self):
        """FIXED: Simple arm position tracking - just pull toward defaults
        
        For standing tasks, we don't need velocity/acceleration penalties on arms.
        Just gently encourage arms to stay near their default position.
        """
        arm_positions = self.dof_pos[:, self.arm_dof_indices]
        arm_default = self.default_dof_pos[:, self.arm_dof_indices]
        
        # Simple squared error - no clamping needed, no velocity terms
        position_error = arm_positions - arm_default
        
        return torch.sum(torch.square(position_error), dim=1)
    
    def _reward_com_position(self):
        """Center of mass over support polygon"""
        # Simplified: penalize CoM deviation from midpoint between feet
        left_foot_pos = self.feet_pos[:, 0, :]
        right_foot_pos = self.feet_pos[:, 1, :]
        midpoint = (left_foot_pos + right_foot_pos) / 2.0
        
        com_xy = self.base_pos[:, :2]
        midpoint_xy = midpoint[:, :2]
        
        return torch.sum(torch.square(com_xy - midpoint_xy), dim=1)
    
    
    
