"""
G1 Standing Task Configuration - Balanced Version v3
Save as: legged_gym/envs/g1/g1_standing_config.py
"""

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1StandingCfg(G1RoughCfg):
    """Configuration for standing balance task"""
    
    class env(G1RoughCfg.env):
        num_envs = 512
        episode_length_s = 20
        num_observations = 47
        num_privileged_obs = 50
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
        pos = [0.0, 0.0, 0.78]
        # Keep default joint positions from base config
    
    class control(G1RoughCfg.control):
        # Balanced control - not too stiff, not too loose
        stiffness = {'joint': 60.0}   # Moderate stiffness
        damping = {'joint': 2.0}      # Moderate damping
        action_scale = 0.5            # Allow reasonable action range
    
    class asset(G1RoughCfg.asset):
        pass
    
    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False
        
        # Standing task parameters - reasonable tolerances for learning
        target_height = 0.78
        height_tolerance = 0.04        # Reasonable tolerance
        orientation_tolerance = 0.2    # ~11.5 degrees - learnable
        velocity_tolerance = 0.1       # Allow some movement for balance
        joint_pos_tolerance = 0.15     # Allow joint movement for balance
        
        class scales(G1RoughCfg.rewards.scales):
            # === PHASE 1: Learn to not fall (most important) ===
            alive = 2.0                    # Strong reward for staying upright
            termination = -2.0             # Strong penalty for falling
            
            # === PHASE 2: Maintain upright posture ===
            orientation_upright = 5.0      # Reward staying upright
            base_height = 3.0              # Reward correct height
            
            # === PHASE 3: Minimize unnecessary movement ===
            base_lin_vel_xy = 1.5          # Prefer staying still
            base_ang_vel_xy = 1.0          # Prefer no rotation
            base_ang_vel_z = 0.5           # Prefer no yaw rotation
            
            # Joint stability (but allow balancing movements)
            default_joint_pos = 0.5        # Gentle preference for default pose
            
            # Feet should stay on ground
            feet_contact = 1.5
            
            # === SMOOTHNESS (secondary priority) ===
            action_rate = -0.01            # Gentle smoothness preference
            dof_vel = -5e-4                # Allow joint movement for balance
            torques = -1e-5                # Allow torques for balance
            
            # === DISABLE LOCOMOTION ===
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            feet_swing_height = 0.0
            
            # === SAFETY PENALTIES (moderate) ===
            lin_vel_z = -2.0               # Penalize bouncing
            ang_vel_xy = -0.2              # Gentle penalty for tilt velocity
            dof_acc = -2.5e-7              # Gentle smoothness
            dof_pos_limits = -10.0         # Strong penalty near limits
            collision = -1.0
            hip_pos = -0.5
            
            # Disabled
            contact = 0.0
            contact_no_vel = 0.0
    
    class domain_rand(G1RoughCfg.domain_rand):
        # Start with minimal randomization, increase as training progresses
        randomize_friction = True
        friction_range = [0.6, 1.2]
        
        randomize_base_mass = True
        added_mass_range = [-0.5, 1.0]
        
        push_robots = True
        push_interval_s = 15           # Infrequent pushes initially
        max_push_vel_xy = 0.2          # Very gentle pushes
        
        randomize_gains = False        # Disable initially for easier learning
        stiffness_multiplier_range = [0.9, 1.1]
        damping_multiplier_range = [0.9, 1.1]
        
        randomize_base_com = False     # Disable for easier learning
    
    class normalization(G1RoughCfg.normalization):
        class obs_scales(G1RoughCfg.normalization.obs_scales):
            lin_vel = 2.0      # Keep default scaling
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
    
    class noise(G1RoughCfg.noise):
        add_noise = True
        noise_level = 0.5  # Moderate noise


class G1StandingCfgPPO(G1RoughCfgPPO):
    """PPO config for standing task"""
    
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 1.0           # Higher exploration initially
        actor_hidden_dims = [128, 64]  # Simpler network learns basics faster
        critic_hidden_dims = [128, 64]
        activation = 'elu'
        
        # RNN for temporal dependencies
        rnn_type = 'lstm'
        rnn_hidden_size = 128
        rnn_num_layers = 1
    
    class algorithm(G1RoughCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.01            # Encourage exploration
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 3e-4           # Standard learning rate
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
        run_name = 'standing'
        experiment_name = 'g1_standing'
        
        # Logging and saving
        save_interval = 100
        log_interval = 10
        
        # Load checkpoint if resuming
        resume = False
        load_run = -1
        checkpoint = -1