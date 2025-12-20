"""
G1 Standing Task Configuration
Save as: legged_gym/envs/g1/g1_standing_config.py
"""

from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1StandingCfg(G1RoughCfg):
    """Configuration for standing balance task"""
    
    class env(G1RoughCfg.env):
        num_envs = 512
        episode_length_s = 20  # 20 seconds per episode
        num_observations = 47  # Keep same as G1RoughCfg
        num_privileged_obs = 50
        num_actions = 12
    
    class terrain(G1RoughCfg.terrain):
        mesh_type = 'plane'  # Flat ground
        measure_heights = False
    
    class commands(G1RoughCfg.commands):
        curriculum = False
        # Keep num_commands = 4 to match base class expectations
        # Commands: [lin_vel_x, lin_vel_y, ang_vel_yaw, heading]
        num_commands = 4
        
        class ranges:
            # Stand still - all commands are zero
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
    
    class init_state(G1RoughCfg.init_state):
        pos = [0.0, 0.0, 0.8]  # Standing height
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
        only_positive_rewards = False  # IMPORTANT: Add this missing attribute
        
        # Custom reward function parameters for standing
        target_height = 0.78           # Match base_height_target
        height_sigma = 0.05            # Tolerance for height
        orientation_sigma = 0.25       # Tolerance for orientation (radians)
        base_motion_sigma = 0.1        # Tolerance for movement
        
        class scales(G1RoughCfg.rewards.scales):
            # === PRIMARY OBJECTIVES (positive rewards) ===
            # Custom standing rewards
            base_height = 3.0              # NEW: Maintain standing height
            orientation = 2.0              # NEW: Stay upright (custom implementation)
            base_stability = 1.0           # NEW: Minimize movement
            
            # Keep alive reward
            alive = 0.5                    # INCREASED: Reward for not falling
            
            # Disable walking rewards
            tracking_lin_vel = 0.0         # CHANGED: Don't track velocity
            tracking_ang_vel = 0.0         # CHANGED: Don't track angular velocity
            feet_air_time = 0.0            # CHANGED: Feet should stay on ground
            contact = 0.0                  # CHANGED: Disable gait contact reward
            feet_swing_height = 0.0        # CHANGED: No swinging
            
            # Penalties from G1RoughCfg (keep these)
            lin_vel_z = -2.0               # Penalize vertical movement
            ang_vel_xy = -0.05             # Penalize roll/pitch rotation
            # Note: We removed the default 'orientation' and 'base_height' penalties
            # and replaced them with positive rewards above
            
            # Energy penalties (keep from G1RoughCfg)
            action_rate = -0.01
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            dof_pos_limits = -5.0
            
            # Safety
            collision = 0.0
            
            # G1 specific penalties (keep)
            hip_pos = -0.5                 # Reduced penalty
            contact_no_vel = 0.0           # Disabled for standing
    
    class domain_rand(G1RoughCfg.domain_rand):
        # Use domain randomization for robustness
        randomize_friction = True
        friction_range = [0.1, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 5
        max_push_vel_xy = 1.0  # Reduced push for standing
    
    class normalization(G1RoughCfg.normalization):
        # Inherit normalization from G1RoughCfg
        pass
    
    class noise(G1RoughCfg.noise):
        # Inherit noise settings from G1RoughCfg
        pass


class G1StandingCfgPPO(G1RoughCfgPPO):
    """PPO config for standing task"""
    
    class policy(G1RoughCfgPPO.policy):
        # Use same network architecture as G1RoughCfg
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
        policy_class_name = "ActorCriticRecurrent"  # Use same as G1RoughCfg
        run_name = 'standing'
        experiment_name = 'g1_standing'
        max_iterations = 1000
        save_interval = 100