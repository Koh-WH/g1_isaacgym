import isaacgym
import isaacgym.gymapi as gymapi

# Initialize gym
gym = gymapi.acquire_gym()

# Create simulator with NO PHYSICS
sim_params = gymapi.SimParams()
sim_params.dt = 1.0/60.0
sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)  # NO GRAVITY
sim_params.up_axis = gymapi.UP_AXIS_Z
sim_params.use_gpu_pipeline = False
sim_params.physx.solver_type = 0  # Disable solver
sim_params.physx.num_position_iterations = 0  # No position iterations
sim_params.physx.num_velocity_iterations = 0  # No velocity iterations

sim = gym.create_sim(0, 0, gymapi.SIM_PHYSX, sim_params)

# Create viewer
viewer = gym.create_viewer(sim, gymapi.CameraProperties())

# Add ground plane (optional)
plane_params = gymapi.PlaneParams()
plane_params.normal = gymapi.Vec3(0, 0, 1)
plane_params.distance = 0
gym.add_ground(sim, plane_params)

# Default joint angles ############################################# Change Joints here #############################################
default_joint_angles = {
    # LEGS
    'left_hip_pitch_joint': -0.23,
    'left_hip_roll_joint': 0.0,
    'left_hip_yaw_joint': 0.0,
    'left_knee_joint': 0.32,
    'left_ankle_pitch_joint': -0.12,
    'left_ankle_roll_joint': 0.0,

    'right_hip_pitch_joint': -0.23,
    'right_hip_roll_joint': 0.0,
    'right_hip_yaw_joint': 0.0,
    'right_knee_joint': 0.32,
    'right_ankle_pitch_joint': -0.12,
    'right_ankle_roll_joint': 0.0,

    # WAIST
    'waist_yaw_joint': 0.0,
    'waist_roll_joint': 0.0,
    'waist_pitch_joint': 0.0,

    # ARMS
    'left_shoulder_pitch_joint': 0.2,
    'left_shoulder_roll_joint': 0.2,
    'left_shoulder_yaw_joint': 0.0,
    'left_elbow_joint': 1.0,
    'left_wrist_roll_joint': 0.0,
    'left_wrist_pitch_joint': 0.0,
    'left_wrist_yaw_joint': 0.0,

    'right_shoulder_pitch_joint': 0.2,
    'right_shoulder_roll_joint': -0.2,
    'right_shoulder_yaw_joint': 0.0,
    'right_elbow_joint': 1.0,
    'right_wrist_roll_joint': 0.0,
    'right_wrist_pitch_joint': 0.0,
    'right_wrist_yaw_joint': 0.0,
}

# Load robot asset - FIXED BASE, NO GRAVITY
asset_root = '/home/koh-wh/Downloads/isaacgym_/unitree_rl_gym/resources/robots/g1_description'
asset_file = 'g1_29dof.urdf'

asset_options = gymapi.AssetOptions()
asset_options.fix_base_link = True          # Base is fixed - won't move
asset_options.disable_gravity = True         # NO gravity effect
asset_options.default_dof_drive_mode = 0     # DOF_MODE_NONE (0) - no physics control
asset_options.use_mesh_materials = True

print("Loading asset...")
robot_asset = gym.load_urdf(sim, asset_root, asset_file, asset_options)

# Get DOF properties
num_dofs = gym.get_asset_dof_count(robot_asset)
dof_names = gym.get_asset_dof_names(robot_asset)

print(f"Number of DOFs: {num_dofs}")
print(f"DOF names: {dof_names}")

# Create environment
env = gym.create_env(sim, gymapi.Vec3(-2.0, 0.0, -2.0), gymapi.Vec3(2.0, 2.0, 2.0), 1)

# Create actor - place at ground level
pose = gymapi.Transform()
pose.p = gymapi.Vec3(0.0, 0.0, 0.85)  # On the ground
pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

actor_handle = gym.create_actor(env, robot_asset, pose, "G1_Robot", 0, 1)

# Set DOF positions to default pose using the CORRECT API
dof_states = gym.get_actor_dof_states(env, actor_handle, gymapi.STATE_ALL)

print("\nSetting default joint angles:")
for i, dof_name in enumerate(dof_names):
    # Find matching angle in default_joint_angles
    angle = 0.0  # default
    
    # Exact match first
    if dof_name in default_joint_angles:
        angle = default_joint_angles[dof_name]
        print(f"  {dof_name}: {angle:.3f} rad (exact match)")
    else:
        # Try partial match
        for default_name, default_angle in default_joint_angles.items():
            # Check if names are similar (case insensitive, ignore underscores)
            if (default_name.replace('_', '').lower() in dof_name.replace('_', '').lower() or
                dof_name.replace('_', '').lower() in default_name.replace('_', '').lower()):
                angle = default_angle
                print(f"  {dof_name} → {default_name}: {angle:.3f} rad (partial match)")
                break
        else:
            # No match found - guess based on joint type
            if 'knee' in dof_name.lower():
                angle = 0.3
            elif 'hip' in dof_name.lower() and 'pitch' in dof_name.lower():
                angle = -0.1
            elif 'ankle' in dof_name.lower() and 'pitch' in dof_name.lower():
                angle = -0.2
            elif 'elbow' in dof_name.lower():
                angle = 0.3
            elif 'shoulder' in dof_name.lower() and 'roll' in dof_name.lower():
                if 'left' in dof_name.lower():
                    angle = 0.15
                elif 'right' in dof_name.lower():
                    angle = -0.15
            print(f"  {dof_name}: {angle:.3f} rad (guessed)")
    
    # Set the position
    dof_states['pos'][i] = angle

# Apply all DOF states at once
gym.set_actor_dof_states(env, actor_handle, dof_states, gymapi.STATE_POS)

# Also set DOF position targets for good measure (if using position control)
dof_props = gym.get_actor_dof_properties(env, actor_handle)
if 'driveMode' in dof_props:
    dof_props['driveMode'][:] = gymapi.DOF_MODE_POS
    gym.set_actor_dof_properties(env, actor_handle, dof_props)

# Set camera view
cam_pos = gymapi.Vec3(2.0, 2.0, 1.0)
cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
gym.viewer_camera_look_at(viewer, None, cam_pos, cam_target)

print("\n" + "="*60)
print("STATIC VISUALIZATION ONLY - NO PHYSICS")
print("Robot is in default pose, frozen in place.")
print("Press 'Q' to quit.")
print("="*60)

# Main loop - NO PHYSICS SIMULATION
frame = 0
while not gym.query_viewer_has_closed(viewer):
    # Optional: Rotate camera slowly for better view
    if frame % 100 == 0:
        # Optional: Print robot state
        body_states = gym.get_actor_rigid_body_states(env, actor_handle, gymapi.STATE_POS)
        base_pos = body_states['pose']['p'][0]
        print(f"Frame {frame}: Base at ({base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f})")
    
    # Just update graphics - NO PHYSICS
    gym.step_graphics(sim)
    gym.draw_viewer(viewer, sim, True)
    gym.sync_frame_time(sim)
    frame += 1

# Cleanup
gym.destroy_viewer(viewer)
gym.destroy_sim(sim)