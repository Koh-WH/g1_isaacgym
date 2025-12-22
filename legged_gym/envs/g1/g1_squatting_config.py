"""
G1 Squatting Task Configuration
Save as: legged_gym/envs/g1/g1_squatting_config.py
"""

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1SquattingCfg(G1RoughCfg):
    """Configuration for continuous squatting task"""
    
    class env(G1RoughCfg.env):
        num_envs = 512
        episode_length_s = 20
        num_observations = 48  # Added squat phase to observations
        num_privileged_obs = 51
        num_actions = 12
    
    class terrain(G1RoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False
    
    class commands(G1RoughCfg.commands):
        curriculum = False
        num_commands = 4
        
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
    
    class init_state(G1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.78]  # Standing height
    
    class control(G1RoughCfg.control):
        stiffness = {'joint': 60.0}
        damping = {'joint': 2.0}
        action_scale = 0.5
    
    class asset(G1RoughCfg.asset):
        pass
    
    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False
        
        # Squatting parameters
        squat_frequency = 0.5          # Hz (0.5 = one squat every 2 seconds)
        standing_height = 0.78         # Standing height
        squat_depth = 0.25            # How much to lower (in meters)
        squat_height = 0.53            # Target height at bottom (0.78 - 0.25)
        height_tolerance = 0.05        # Tolerance for height tracking
        orientation_tolerance = 0.2    # Stay upright during squats
        
        # Hip and knee joint indices (adjust based on your robot's URDF)
        # Typically: left_hip_pitch, right_hip_pitch, left_knee, right_knee
        hip_joints = [0, 6]           # Adjust these indices!
        knee_joints = [1, 7]          # Adjust these indices!
        
        class scales(G1RoughCfg.rewards.scales):
            # === PRIMARY SQUATTING OBJECTIVES ===
            # Track target height (squatting motion)
            track_squat_height = 8.0       # Main objective: follow squat trajectory
            
            # Maintain upright orientation during squats
            orientation_upright = 5.0      
            
            # Proper joint configuration for squatting
            hip_knee_coordination = 3.0    # Reward proper hip/knee bending
            
            # Keep both feet planted during squats
            feet_contact = 4.0
            
            # Smooth squatting motion
            smooth_height_change = 2.0     # Reward smooth transitions
            
            # === STABILITY ===
            base_lin_vel_xy = 2.0          # Don't move horizontally
            base_ang_vel_xy = 1.5          # Don't rotate while squatting
            base_ang_vel_z = 0.5
            
            # Survival
            alive = 2.0
            termination = -2.0
            
            # === DISABLE LOCOMOTION ===
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            feet_swing_height = 0.0
            
            # === PENALTIES ===
            # Control smoothness
            action_rate = -0.02
            dof_vel = -5e-4
            torques = -2e-5
            dof_acc = -2.5e-7
            
            # Horizontal motion penalties
            lin_vel_z = -1.0               # Allow some vertical motion for squats
            ang_vel_xy = -0.2
            
            # Safety
            dof_pos_limits = -10.0
            collision = -1.0
            hip_pos = -0.3
            
            # Disabled
            contact = 0.0
            contact_no_vel = 0.0
    
    class domain_rand(G1RoughCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.6, 1.2]
        
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.0]
        
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 0.2
        
        randomize_gains = False
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
    
    class normalization(G1RoughCfg.normalization):
        class obs_scales(G1RoughCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
    
    class noise(G1RoughCfg.noise):
        add_noise = True
        noise_level = 0.5


class G1SquattingCfgPPO(G1RoughCfgPPO):
    """PPO config for squatting task"""
    
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64]
        critic_hidden_dims = [128, 64]
        activation = 'elu'
        
        rnn_type = 'lstm'
        rnn_hidden_size = 128
        rnn_num_layers = 1
    
    class algorithm(G1RoughCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3e-4
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0
    
    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 3000
        run_name = 'squatting'
        experiment_name = 'g1_squatting'
        
        save_interval = 100
        log_interval = 10
        
        resume = False
        load_run = -1
        checkpoint = -1