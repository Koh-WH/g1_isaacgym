#!/usr/bin/env python3
"""
G1 Robot: Hybrid Control - FIXED for g1_standing_config.py
===========================================================
- Matches training configuration exactly
- Corrected PD gains, initial height, ankle pitch, and commands
- Usage: 
python keyboard_g1_policy_fixed.py --robot=/path/to/g1_29dof.urdf --policy=/path/to/policy_1.pt
"""

import argparse
import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import os
import sys
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# ===================================================================
# 🛠️ USER CONFIGURATION: MATCHES g1_standing_config.py
# ===================================================================
USER_SPAWN_POS = {
    # === LEGS (CORRECTED ankle pitch: -0.25) ===
    'left_hip_pitch_joint': -0.15, 
    'left_hip_roll_joint': 0.0, 
    'left_hip_yaw_joint': 0.0,
    'left_knee_joint': 0.4, 
    'left_ankle_pitch_joint': -0.25,  # ✅ FIXED: was -0.15
    'left_ankle_roll_joint': 0.0,
    
    'right_hip_pitch_joint': -0.15, 
    'right_hip_roll_joint': 0.0, 
    'right_hip_yaw_joint': 0.0,
    'right_knee_joint': 0.4, 
    'right_ankle_pitch_joint': -0.25,  # ✅ FIXED: was -0.15
    'right_ankle_roll_joint': 0.0,
    
    # === ARMS (unchanged) ===
    'left_shoulder_pitch_joint': 0.2, 
    'left_shoulder_roll_joint': 0.2, 
    'left_elbow_joint': 1.0,
    'right_shoulder_pitch_joint': 0.2, 
    'right_shoulder_roll_joint': -0.2, 
    'right_elbow_joint': 1.0,
}
# ===================================================================

class G1KeyboardHybrid:
    def __init__(self, args):
        self.args = args
        
        # 1. Physics Setup FIRST
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0/50.0  # Matches decimation=10 with dt=0.002
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.use_gpu_pipeline = True  
        sim_params.physx.use_gpu = True
        
        # 2. Set device based on GPU pipeline setting
        self.device = "cuda:0" if sim_params.use_gpu_pipeline else "cpu"
        print(f"🚀 Initializing on {self.device}...")
        
        self.running = True
        self.gym = gymapi.acquire_gym()
        
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        if self.sim is None: sys.exit("❌ Failed to create sim.")

        self._create_ground()
        self.robot_handle = self._load_robot()
        self._setup_viewer()
        
        self.gym.prepare_sim(self.sim)
        self._acquire_tensors()
        
        # 2. Policy Setup
        self.policy_path = args.policy
        self.policy_input_dim = 47
        self.policy_output_dim = 12
        self.load_policy_and_detect_dims()
        
        self.rl_active_dofs = 12  # RL controls legs only
        
        # 3. Init Controls
        self.rl_enabled = True        
        self.control_enabled = True   
        self.reverse_mode = False
        self.lock_base = False 
        
        self.default_dof_pos = torch.zeros(self.num_dofs, device=self.device)
        self.init_default_pose() 
        self.target_pos = self.default_dof_pos.clone()
        
        # ✅ FIXED: Commands match training (forward velocity 0.5 m/s)
        self.commands = torch.tensor([0.5, 0.0, 0.0], device=self.device)
        self.last_actions = torch.zeros(self.policy_output_dim, device=self.device)
        
        self.print_controls()
        print(f"\n✅ Configuration matched to g1_standing_config.py")
        print(f"   - Initial height: 1.0m")
        print(f"   - Hip stiffness: 100.0, Knee: 150.0, Ankle: 40.0")
        print(f"   - Command: Forward 0.5 m/s\n")

    def print_controls(self):
        print("\n" + "="*60)
        print("🎮 G1 CONTROL MAPPING")
        print("="*60)
        print("  [L]          : 🔒 LOCK/UNLOCK HIPS (Hover in place)")
        print("  [R]          : 🔄 Respawn Robot")
        print("  [ENTER]      : 🧠 Toggle RL Balance (Legs)")
        print("  [SPACE]      : 🖐️ Toggle Manual Control (Upper Body)")
        print("  [BACKSPACE]  : ↩️ Reset Manual Offsets")
        print("  [+/-]        : ↔️ Change Joint Direction (Forward/Reverse)")
        print("-" * 60)
        print("  [1-6]        : Left Leg   | [7-0, Q,W] : Right Leg")
        print("  [E,Z,T]      : Waist      |")
        print("  [Y,U,I,O]    : Left Arm   | [D,F,G,H]  : Right Arm")
        print("  [P,A,S]      : Left Hand  | [J,K,L]    : Right Hand")
        print("="*60 + "\n")

    def _create_ground(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)

    def _load_robot(self):
        user_input = self.args.robot
        candidates = [user_input]
        if not (user_input.endswith('.xml') or user_input.endswith('.urdf')):
            candidates.append(f"{user_input}.xml")
            candidates.append(f"{user_input}.urdf")
            
        search_dirs = [
            os.getcwd(),
            os.path.join(os.getcwd(), "resources", "robots", "g1_description"),
            os.path.join(os.path.expanduser("~"), "Downloads"),
            os.path.dirname(user_input)
        ]
        
        final_path = None
        for c in candidates:
            if os.path.exists(c): 
                final_path = c
                break
            for d in search_dirs:
                test_path = os.path.join(d, os.path.basename(c))
                if os.path.exists(test_path): 
                    final_path = test_path
                    break
            if final_path: break

        if not final_path: 
            sys.exit(f"❌ Robot not found: {user_input}")
        
        print(f"📂 Loading: {final_path}")

        asset_opts = gymapi.AssetOptions()
        asset_opts.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        
        robot_asset = self.gym.load_asset(
            self.sim, 
            os.path.dirname(final_path), 
            os.path.basename(final_path), 
            asset_opts
        )
        
        self.num_dofs = self.gym.get_asset_dof_count(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        
        env_lower = gymapi.Vec3(-2.0, -2.0, 0.0)
        env_upper = gymapi.Vec3(2.0, 2.0, 2.0)
        self.env = self.gym.create_env(self.sim, env_lower, env_upper, 1)
        
        # ✅ FIXED: Initial height 1.0m (was 0.82)
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(0.0, 0.0, 1.0)
        handle = self.gym.create_actor(self.env, robot_asset, pose, "g1", 0, 1)
        
        # ✅ FIXED: PD Gains matching g1_standing_config.py
        props = self.gym.get_actor_dof_properties(self.env, handle)
        
        for i, name in enumerate(self.dof_names):
            if 'hip' in name:
                props['stiffness'][i] = 100.0  # Training: 100.0
                props['damping'][i] = 2.0      # Training: 2.0
            elif 'knee' in name:
                props['stiffness'][i] = 150.0  # Training: 150.0
                props['damping'][i] = 4.0      # Training: 4.0
            elif 'ankle' in name:
                props['stiffness'][i] = 40.0   # Training: 40.0
                props['damping'][i] = 2.0      # Training: 2.0
            else:
                # Upper body (not controlled by this policy)
                props['stiffness'][i] = 40.0
                props['damping'][i] = 2.0
        
        self.gym.set_actor_dof_properties(self.env, handle, props)
        
        return handle

    def _acquire_tensors(self):
        root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.root_states = gymtorch.wrap_tensor(root_tensor)
        self.dof_states = gymtorch.wrap_tensor(dof_tensor)
        self.dof_pos = self.dof_states[:, 0]
        self.dof_vel = self.dof_states[:, 1]
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)

    def load_policy_and_detect_dims(self):
        if not os.path.exists(self.policy_path): 
            sys.exit(f"❌ Policy not found: {self.policy_path}")
        try:
            self.policy = torch.jit.load(self.policy_path, map_location=self.device)
            self.policy.eval()
            detected = False
            for dim in range(40, 150):
                try:
                    dummy = torch.zeros(1, dim, device=self.device)
                    with torch.no_grad(): 
                        out = self.policy(dummy)
                    self.policy_input_dim = dim
                    self.policy_output_dim = out.shape[1]
                    detected = True
                    print(f"✅ Policy detected: {dim} inputs → {out.shape[1]} outputs")
                    break
                except: 
                    continue
            if not detected: 
                print("⚠️ Auto-detect failed. Using defaults: 47 → 12")
                self.policy_output_dim = 12
        except Exception as e: 
            sys.exit(f"❌ Error loading policy: {e}")

    def init_default_pose(self):
        pose = np.zeros(self.num_dofs)
        print("🔧 Applying Custom Spawn Pose...")
        for name, angle in USER_SPAWN_POS.items():
            try:
                idx = self.dof_names.index(name)
                pose[idx] = angle
            except ValueError: 
                pass
        self.default_dof_pos = torch.tensor(pose, device=self.device, dtype=torch.float)

    def reset_game(self):
        print("🔄 RESPAWNING...")
        self.root_states.fill_(0.0)
        self.root_states[0, 2] = 1.0  # ✅ FIXED: height 1.0m
        self.root_states[0, 6] = 1.0
        self.dof_pos[:] = self.default_dof_pos
        self.dof_vel.fill_(0.0)
        self.target_pos = self.default_dof_pos.clone()
        self.commands = torch.tensor([0.5, 0.0, 0.0], device=self.device)  # Reset to forward
        self.last_actions.fill_(0.0)
        self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))

    def compute_observations(self):
        N = self.policy_output_dim
        base_quat = self.root_states[0, 3:7] / torch.norm(self.root_states[0, 3:7])
        r = R.from_quat(base_quat.cpu().numpy())
        proj_grav = torch.tensor(r.inv().apply([0, 0, -1]), device=self.device, dtype=torch.float)
        base_ang_vel = self.root_states[0, 10:13] * 0.25
        cmds = self.commands * torch.tensor([2.0, 2.0, 0.25], device=self.device)
        
        dof_pos_scaled = (self.dof_pos[:N] - self.default_dof_pos[:N]) * 1.0
        dof_vel_scaled = self.dof_vel[:N] * 0.05
        
        obs = torch.cat([base_ang_vel, proj_grav, cmds, dof_pos_scaled, dof_vel_scaled, self.last_actions])
        
        if obs.shape[0] < self.policy_input_dim:
            obs = torch.nn.functional.pad(obs, (0, self.policy_input_dim - obs.shape[0]))
        elif obs.shape[0] > self.policy_input_dim:
            obs = obs[:self.policy_input_dim]
        return obs.unsqueeze(0)

    def _setup_viewer(self):
        self.viewer = self.gym.create_viewer(self.sim, gymapi.CameraProperties())
        cam_pos = gymapi.Vec3(2.0, -2.0, 1.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, gymapi.Vec3(0, 0, 0.8))
        
        # KEY BINDINGS
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_L, "lock")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_ENTER, "enter")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_BACKSPACE, "backspace")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_EQUAL, "plus")
        
        keys = ['1','2','3','4','5','6','7','8','9','0',
                'q','w','e','z','t','y','u','i','o','p',
                'a','s','d','f','g','h','j','k','l']
        
        self.joint_map = {}
        for idx, key in enumerate(keys[:self.num_dofs]):
            self.joint_map[key] = idx
            if key.isdigit(): 
                key_code = getattr(gymapi, f'KEY_{key}')
            else: 
                key_code = getattr(gymapi, f'KEY_{key.upper()}')
            self.gym.subscribe_viewer_keyboard_event(self.viewer, key_code, key)

    def handle_keyboard(self):
        step = 0.15 
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value <= 0: continue
            
            if evt.action == "reset": 
                self.reset_game()
                continue
                
            if evt.action == "lock":
                self.lock_base = not self.lock_base
                print(f"🔒 Hip Lock: {'ON' if self.lock_base else 'OFF'}")
            
            if evt.action == "enter":
                self.rl_enabled = not self.rl_enabled
                print(f"🧠 RL Balance: {'ON' if self.rl_enabled else 'OFF'}")
                
            if evt.action == "space":
                self.control_enabled = not self.control_enabled
                print(f"🎮 Manual Control: {'ENABLED' if self.control_enabled else 'LOCKED'}")
                
            if evt.action == "minus":
                self.reverse_mode = not self.reverse_mode
                print(f"🔄 Reverse Mode: {'ON' if self.reverse_mode else 'OFF'}")
                
            if evt.action == "plus": 
                self.reverse_mode = False
                print("⬆️ Forward Mode")
                
            if evt.action == "backspace":
                start_idx = self.rl_active_dofs
                self.target_pos[start_idx:] = self.default_dof_pos[start_idx:]
                print("↩️ Reset Manual Offsets")

            if self.control_enabled and evt.action in self.joint_map:
                idx = self.joint_map[evt.action]
                if self.rl_enabled and idx < self.rl_active_dofs:
                    # Skip leg joints controlled by RL
                    pass
                else:
                    direction = -1 if self.reverse_mode else 1
                    self.target_pos[idx] += step * direction
                    print(f"   Moved {self.dof_names[idx]} ({idx}) -> {self.target_pos[idx].item():.2f}")

    def run(self):
        while not self.gym.query_viewer_has_closed(self.viewer):
            self.handle_keyboard()
            
            # --- HIP LOCK FEATURE ---
            if self.lock_base:
                self.root_states[0, 7:13] = 0.0  # Kill velocity
                self.root_states[0, 2] = 1.0     # ✅ FIXED: Force height to 1.0m
                self.root_states[0, 3:7] = torch.tensor([0, 0, 0, 1], device=self.device)  # Upright
                self.gym.set_actor_root_state_tensor(self.sim, gymtorch.unwrap_tensor(self.root_states))

            if self.rl_enabled:
                self.gym.refresh_actor_root_state_tensor(self.sim)
                self.gym.refresh_dof_state_tensor(self.sim)
                
                obs = self.compute_observations()
                with torch.no_grad(): 
                    actions = self.policy(obs)[0]
                self.last_actions = actions
                
                N_legs = self.rl_active_dofs
                self.target_pos[:N_legs] = actions[:N_legs] * 0.25 + self.default_dof_pos[:N_legs]

            target_tensor = gymtorch.unwrap_tensor(self.target_pos.unsqueeze(0))
            self.gym.set_dof_position_target_tensor(self.sim, target_tensor)
            
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.step_graphics(self.sim)
            self.gym.draw_viewer(self.viewer, self.sim, True)
            self.gym.sync_frame_time(self.sim)

        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=str, required=True, help="Path to .pt policy")
    parser.add_argument("--robot", type=str, required=True, help="Robot file (e.g. g1_29dof.xml)")
    args, unknown = parser.parse_known_args()
    sim = G1KeyboardHybrid(args)
    sim.run()