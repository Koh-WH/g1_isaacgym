#!/usr/bin/env python3
"""
Minimal G1 Robot Keyboard Control in Isaac Gym

SPACE      - Toggle control ON/OFF
BACKSPACE  - Reset all joints to 0
R          - Reset environment (robot, ball, table)
ESC/Close  - Quit simulation
```

### **Direction Controls**
```
MINUS (-)  - Toggle REVERSE mode ⬇️ (keys decrease angles)
PLUS (+)   - Set FORWARD mode ⬆️ (keys increase angles, default)
```

### **Joint Controls** (All 29 Joints)

#### **LEFT LEG** 🦵
1 - left_hip_pitch_joint
2 - left_hip_roll_joint
3 - left_hip_yaw_joint
4 - left_knee_joint
5 - left_ankle_pitch_joint
6 - left_ankle_roll_joint

#### **RIGHT LEG** 🦵
7 - right_hip_pitch_joint
8 - right_hip_roll_joint
9 - right_hip_yaw_joint
0 - right_knee_joint
Q - right_ankle_pitch_joint
W - right_ankle_roll_joint

#### **WAIST** 🔄
E - waist_yaw_joint (twist)
Z - waist_roll_joint (lean sideways)
T - waist_pitch_joint (lean forward/back)

#### **LEFT ARM** 💪
Y - left_shoulder_pitch_joint (forward/back)
U - left_shoulder_roll_joint (up/down)
I - left_shoulder_yaw_joint (twist)
O - left_elbow_joint (bend)

#### **LEFT HAND** ✋
P - left_wrist_roll_joint (twist)
A - left_wrist_pitch_joint (up/down)
S - left_wrist_yaw_joint (side to side)

#### **RIGHT ARM** 💪
D - right_shoulder_pitch_joint (forward/back)
F - right_shoulder_roll_joint (up/down)
G - right_shoulder_yaw_joint (twist)
H - right_elbow_joint (bend)

#### **RIGHT HAND** ✋
J - right_wrist_roll_joint (twist)
K - right_wrist_pitch_joint (up/down)
L - right_wrist_yaw_joint (side to side)

## 📋 Quick Start Guide
1. **Start simulation** - Robot stands still
2. **Press SPACE** - Enable control (see "🎮 CONTROL ENABLED")
3. **Press Y** - Left shoulder moves up ⬆️
4. **Press MINUS** - Switch to reverse mode ⬇️
5. **Press Y** - Left shoulder moves down ⬇️
6. **Press O** - Left elbow bends (direction depends on mode)
7. **Press BACKSPACE** - All joints return to 0
8. **Press R** - Reset everything

## 💡 Pro Tips
- **For precise control**: Use MINUS to toggle direction instead of trying to "undo"
- **Arm joints work best**: Y, U, I, O (left) and D, F, G, H (right)
- **Legs locked by high stiffness**: But you can still move them with 1-6, 7-0, Q, W
- **Watch terminal**: Shows current joint angle and direction arrow (⬆️/⬇️)

"""

import isaacgym
from isaacgym import gymapi, gymutil, gymtorch
import os
import sys
import numpy as np
import torch

class G1KeyboardSim:
    """Minimal G1 robot with keyboard control"""
    
    def __init__(self):
        self.running = True
        self.num_dofs = 0
        self.dof_names = []
        self.target_positions = None
        self.control_enabled = False
        self.reverse_mode = False  # Forward/Reverse direction toggle
        
        # Create simulation
        self.gym = gymapi.acquire_gym()
        sim_params = self._setup_physics()
        self.sim = self.gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)
        
        # Create ground
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0, 0, 1)
        self.gym.add_ground(self.sim, plane_params)
        
        # Load robot
        self._load_robot()
        
        # Create environment
        self._create_environment()
        
        # Setup viewer and keyboard
        self._setup_viewer()
        
        # Get tensors
        self._acquire_tensors()
        
        print("\n✅ Simulation ready!")
        print("Press SPACE to enable control, then use mapped keys to move joints\n")
    
    def _setup_physics(self):
        """Setup physics parameters"""
        sim_params = gymapi.SimParams()
        sim_params.dt = 1.0/60.0
        sim_params.substeps = 2
        sim_params.up_axis = gymapi.UP_AXIS_Z
        sim_params.gravity = gymapi.Vec3(0.0, 0.0, -9.81)
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 1
        return sim_params
    
    def _load_robot(self):
        """Load G1 robot URDF"""
        # Find URDF file
        possible_paths = [
            os.path.join(os.path.expanduser("~"), "Downloads", "isaacgym_", 
                        "unitree_rl_gym", "resources", "robots", 
                        "g1_description", "g1_29dof.urdf"),
            "resources/robots/g1_description/g1_29dof.urdf"
        ]
        
        urdf_path = None
        for path in possible_paths:
            if os.path.exists(path):
                urdf_path = path
                break
        
        if not urdf_path:
            print("❌ G1 URDF not found! Place it in one of these locations:")
            for path in possible_paths:
                print(f"   - {path}")
            sys.exit(1)
        
        # Load asset
        asset_root = os.path.dirname(os.path.dirname(urdf_path))
        asset_file = "g1_description/g1_29dof.urdf"
        
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True  # Keep robot standing
        asset_options.default_dof_drive_mode = int(gymapi.DOF_MODE_POS)
        
        self.robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dofs = self.gym.get_asset_dof_count(self.robot_asset)
        
        # Get joint names
        for i in range(self.num_dofs):
            self.dof_names.append(self.gym.get_asset_dof_name(self.robot_asset, i))
        
        print(f"✅ Loaded G1 with {self.num_dofs} joints")
    
    def _create_environment(self):
        """Create environment with robot"""
        spacing = 3.0
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        
        self.env = self.gym.create_env(self.sim, lower, upper, 1)
        
        # Add robot
        robot_pose = gymapi.Transform()
        robot_pose.p = gymapi.Vec3(0.0, 0.0, 0.85)
        robot_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        
        self.robot_handle = self.gym.create_actor(
            self.env, self.robot_asset, robot_pose, "g1_robot", 0, 0)
        
        # Set joint properties
        dof_props = self.gym.get_actor_dof_properties(self.env, self.robot_handle)
        for i in range(self.num_dofs):
            dof_props['driveMode'][i] = int(gymapi.DOF_MODE_POS)
            dof_props['stiffness'][i] = 100.0
            dof_props['damping'][i] = 10.0
            dof_props['effort'][i] = 500.0
        
        self.gym.set_actor_dof_properties(self.env, self.robot_handle, dof_props)
    
    def _setup_viewer(self):
        """Setup viewer and keyboard mappings"""
        cam_props = gymapi.CameraProperties()
        self.viewer = self.gym.create_viewer(self.sim, cam_props)
        
        # Position camera
        cam_pos = gymapi.Vec3(2.0, -2.0, 1.5)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.8)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)
        
        # Subscribe to control keys
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_SPACE, "space")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_BACKSPACE, "backspace")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_R, "reset")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_MINUS, "minus")
        self.gym.subscribe_viewer_keyboard_event(self.viewer, gymapi.KEY_EQUAL, "plus")
        
        # Map keys to joints (exact order from specification)
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
        
        # Print mapping with emojis
        print("\n" + "="*80)
        print("🤖 G1 ROBOT - ALL 29 JOINTS MAPPED TO KEYBOARD")
        print("="*80)
        print("\n🎮 CONTROLS:")
        print("="*80)
        print("  SPACE      - Toggle control ON/OFF")
        print("  BACKSPACE  - Reset all joints to 0")
        print("  R          - Reset environment")
        print("  MINUS (-)  - Toggle REVERSE mode ⬇️ (keys decrease angles)")
        print("  PLUS (+)   - Set FORWARD mode ⬆️ (keys increase angles, default)")
        print("="*80)
        
        print("\n### JOINT CONTROLS (All 29 Joints)")
        print("\n#### LEFT LEG 🦵")
        print("1 - left_hip_pitch_joint")
        print("2 - left_hip_roll_joint")
        print("3 - left_hip_yaw_joint")
        print("4 - left_knee_joint")
        print("5 - left_ankle_pitch_joint")
        print("6 - left_ankle_roll_joint")
        
        print("\n#### RIGHT LEG 🦵")
        print("7 - right_hip_pitch_joint")
        print("8 - right_hip_roll_joint")
        print("9 - right_hip_yaw_joint")
        print("0 - right_knee_joint")
        print("Q - right_ankle_pitch_joint")
        print("W - right_ankle_roll_joint")
        
        print("\n#### WAIST 🔄")
        print("E - waist_yaw_joint (twist)")
        print("Z - waist_roll_joint (lean sideways)")
        print("T - waist_pitch_joint (lean forward/back)")
        
        print("\n#### LEFT ARM 💪")
        print("Y - left_shoulder_pitch_joint (forward/back)")
        print("U - left_shoulder_roll_joint (up/down)")
        print("I - left_shoulder_yaw_joint (twist)")
        print("O - left_elbow_joint (bend)")
        
        print("\n#### LEFT HAND ✋")
        print("P - left_wrist_roll_joint (twist)")
        print("A - left_wrist_pitch_joint (up/down)")
        print("S - left_wrist_yaw_joint (side to side)")
        
        print("\n#### RIGHT ARM 💪")
        print("D - right_shoulder_pitch_joint (forward/back)")
        print("F - right_shoulder_roll_joint (up/down)")
        print("G - right_shoulder_yaw_joint (twist)")
        print("H - right_elbow_joint (bend)")
        
        print("\n#### RIGHT HAND ✋")
        print("J - right_wrist_roll_joint (twist)")
        print("K - right_wrist_pitch_joint (up/down)")
        print("L - right_wrist_yaw_joint (side to side)")
        
        print("\n" + "="*80)
        print("📍 Current mode: ⬆️ FORWARD")
        print("="*80 + "\n")
    
    def _acquire_tensors(self):
        """Get state tensors"""
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        self.dof_states = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_states[:, 0].view(1, self.num_dofs)
        self.dof_vel = self.dof_states[:, 1].view(1, self.num_dofs)
        
        self.target_positions = self.dof_pos[0].cpu().numpy().copy()
    
    def reset_robot(self):
        """Reset robot to neutral pose"""
        self.target_positions.fill(0.0)
        self.dof_pos.fill_(0.0)
        self.dof_vel.fill_(0.0)
        
        actions = torch.from_numpy(self.target_positions).float().unsqueeze(0)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))
        self.gym.set_dof_state_tensor(self.sim, gymtorch.unwrap_tensor(self.dof_states))
        
        print("\n🔄 Environment reset - All joints at 0\n")
    
    def handle_keyboard(self):
        """Process keyboard input"""
        step = 0.15  # Movement increment (radians)
        
        for evt in self.gym.query_viewer_action_events(self.viewer):
            if evt.value <= 0:  # Only on key press
                continue
            
            if evt.action == "space":
                self.control_enabled = not self.control_enabled
                if self.control_enabled:
                    self.target_positions = self.dof_pos[0].cpu().numpy().copy()
                    mode_str = "⬇️ REVERSE" if self.reverse_mode else "⬆️ FORWARD"
                    print(f"\n🎮 CONTROL ENABLED - Mode: {mode_str}")
                    print("   Use keys to move joints\n")
                else:
                    print("\n🔒 CONTROL DISABLED - Robot holding current position\n")
            
            elif evt.action == "minus":
                self.reverse_mode = not self.reverse_mode
                mode_str = "⬇️ REVERSE" if self.reverse_mode else "⬆️ FORWARD"
                print(f"\n🔄 Direction mode: {mode_str}")
                print(f"   Keys will now {'DECREASE' if self.reverse_mode else 'INCREASE'} joint angles\n")
            
            elif evt.action == "plus":
                self.reverse_mode = False
                print("\n⬆️ FORWARD mode activated")
                print("   Keys will INCREASE joint angles\n")
            
            elif evt.action == "backspace":
                self.target_positions.fill(0.0)
                print("\n↩️  ALL JOINTS RESET TO 0\n")
            
            elif evt.action == "reset":
                self.reset_robot()
            
            elif evt.action in self.joint_map and self.control_enabled:
                joint_idx = self.joint_map[evt.action]
                direction = -1 if self.reverse_mode else 1
                self.target_positions[joint_idx] += step * direction
                
                arrow = "⬇️" if self.reverse_mode else "⬆️"
                print(f"{arrow} [{joint_idx:2d}] {self.dof_names[joint_idx]:30s} = {self.target_positions[joint_idx]:6.2f} rad")
    
    def step(self):
        """Step simulation"""
        # Handle keyboard
        self.handle_keyboard()
        
        # Apply targets
        actions = torch.from_numpy(self.target_positions).float().unsqueeze(0)
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(actions))
        
        # Simulate
        self.gym.simulate(self.sim)
        self.gym.fetch_results(self.sim, True)
        
        # Refresh states
        self.gym.refresh_dof_state_tensor(self.sim)
        
        # Render
        self.gym.step_graphics(self.sim)
        self.gym.draw_viewer(self.viewer, self.sim, True)
        
        # Check if viewer closed
        if self.gym.query_viewer_has_closed(self.viewer):
            self.running = False
    
    def run(self):
        """Main loop"""
        print("="*60)
        print("🚀 Starting G1 Robot Keyboard Control...")
        print("="*60 + "\n")
        
        try:
            while self.running:
                self.step()
                self.gym.sync_frame_time(self.sim)
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        self.gym.destroy_viewer(self.viewer)
        self.gym.destroy_sim(self.sim)
        print("\n✅ Simulation closed")

if __name__ == "__main__":
    sim = G1KeyboardSim()
    sim.run()