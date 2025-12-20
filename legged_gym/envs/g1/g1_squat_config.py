"""
G1 Squat Task Configuration
Save as: legged_gym/envs/g1/g1_squat_config.py
"""

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1SquatCfg(G1RoughCfg):
    """Configuration for squat exercise task"""
    
    class env(G1RoughCfg.env):
        num_envs = 512
        episode_length_s = 30  # 30 seconds - enough for multiple squats
        num_observations = 47  # Keep same as G1RoughCfg
        num_privileged_obs = 50
        num_actions = 12
    
    class terrain(G1RoughCfg.terrain):
        mesh_type = 'plane'  # Flat ground
        measure_heights = False
    
    class commands(G1RoughCfg.commands):
        curriculum = False
        num_commands = 4
        
        class ranges:
            # No horizontal movement during squats
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
    
    class init_state(G1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # Start standing
        # Use default joint angles from G1RoughCfg
    
    class control(G1RoughCfg.control):
        # Inherit PD control parameters from G1RoughCfg
        pass
    
    class asset(G1RoughCfg.asset):
        # Inherit G1 URDF settings
        pass
    
    class rewards(G1RoughCfg.rewards):
        # Inherit base reward settings
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False
        
        # Squat-specific parameters
        stand_height = 0.78        # Standing position height
        squat_height = 0.45        # Bottom of squat (adjust based on robot limits)
        squat_frequency = 0.5      # 0.5 Hz = one squat every 2 seconds
        height_tolerance = 0.08    # Tolerance for height tracking
        balance_tolerance = 0.3    # Tolerance for orientation
        
        class scales(G1RoughCfg.rewards.scales):
            # === PRIMARY OBJECTIVES (positive rewards) ===
            squat_tracking = 5.0       # NEW: Follow the squat height trajectory
            balance = 2.0              # NEW: Maintain balance during squats
            smooth_motion = 1.0        # NEW: Smooth vertical motion
            feet_contact_squat = 1.5   # NEW: Keep both feet on ground
            
            # Keep alive reward
            alive = 0.3
            
            # Disable walking rewards
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            contact = 0.0
            feet_swing_height = 0.0
            
            # Penalties for unwanted movements
            lin_vel_z = 0.0            # Allow vertical movement for squats
            ang_vel_xy = -0.1          # Penalize roll/pitch rotation
            lateral_movement = -2.0    # NEW: Penalize XY movement
            
            # Energy penalties
            action_rate = -0.01
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            dof_pos_limits = -5.0
            
            # Safety
            collision = 0.0
            
            # G1 specific
            hip_pos = -0.3             # Reduced penalty (hips need to move for squats)
            contact_no_vel = 0.0
    
    class domain_rand(G1RoughCfg.domain_rand):
        # Use domain randomization for robustness
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 8  # Less frequent pushes during squats
        max_push_vel_xy = 0.5  # Gentler pushes
    
    class normalization(G1RoughCfg.normalization):
        pass
    
    class noise(G1RoughCfg.noise):
        pass


class G1SquatCfgPPO(G1RoughCfgPPO):
    """PPO config for squat task"""
    
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 0.8
        actor_hidden_dims = [32]
        critic_hidden_dims = [32]
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 64
        rnn_num_layers = 1
    
    class algorithm(G1RoughCfgPPO.algorithm):
        entropy_coef = 0.01
    
    class runner(G1RoughCfgPPO.runner):
        policy_class_name = "ActorCriticRecurrent"
        run_name = 'squat'
        experiment_name = 'g1_squat'
        max_iterations = 2000  # More iterations for complex motion
        save_interval = 100