from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1StandingCfg(G1RoughCfg):
    """Configuration for standing balance task"""
    
    class env(G1RoughCfg.env):
        num_envs = 512
        episode_length_s = 60.0
        num_observations = 47 
        num_privileged_obs = 50
        num_actions = 12

    class sim(G1RoughCfg.sim):
        dt = 0.002

    class terrain(G1RoughCfg.terrain):
        mesh_type = 'plane'
        measure_heights = False

    class commands(G1RoughCfg.commands):
        curriculum = False
        num_commands = 4 
        
        class ranges:
            lin_vel_x = [0.5, 0.5]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]    
    
    class init_state(G1RoughCfg.init_state):
        pos = [0.0, 0.0, 1.0] 
        default_joint_angles = {
            'left_hip_pitch_joint': -0.1,
            'left_hip_roll_joint': 0.0, 
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.3,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.1,
            'right_hip_roll_joint': 0.0, 
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.3,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
        }
    
    class control(G1RoughCfg.control):
        decimation = 10
        action_scale = 0.25

        stiffness = {
            'hip': 100.0, 
            'knee': 150.0, 
            'ankle': 40.0
        }
        damping = {
            'hip': 2.0, 
            'knee': 4.0, 
            'ankle': 2.0
        }

    class asset(G1RoughCfg.asset):
        pass
    
    class rewards(G1RoughCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False
        
        target_height = 0.78
        height_tolerance = 0.04
        orientation_tolerance = 0.2
        velocity_tolerance = 0.3
        joint_pos_tolerance = 0.8
        
        class scales(G1RoughCfg.rewards.scales):
            alive = 2.0
            termination = -2.0
            orientation_upright = 5.0
            base_height = 3.0
            
            base_lin_vel_xy = 5.0
            base_ang_vel_xy = 2.0
            base_ang_vel_z = 1.0
            
            default_joint_pos = 2.0
            feet_contact = 1.5
            action_rate = -0.2
            dof_vel = -0.05
            torques = -1e-5
            
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            feet_swing_height = 0.0
            
            lin_vel_z = -2.0
            ang_vel_xy = -0.2
            dof_acc = -2.5e-7
            dof_pos_limits = -10.0
            collision = -1.0
            hip_pos = -0.5
            
            contact = 0.0
            contact_no_vel = 0.0
    
    class domain_rand(G1RoughCfg.domain_rand):
            randomize_friction = True
            friction_range = [0.2, 1.5]
            randomize_base_mass = True
            added_mass_range = [-1.0, 5.0] 
            push_robots = True
            push_interval_s = 5
            max_push_vel_xy = 1.0
            randomize_gains = False         
            stiffness_multiplier_range = [0.8, 1.2]
            damping_multiplier_range = [0.8, 1.2]
            randomize_base_com = False     
            added_com_range = [-0.2, 0.2]
    
    class normalization(G1RoughCfg.normalization):
        class obs_scales(G1RoughCfg.normalization.obs_scales):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
    
    class noise(G1RoughCfg.noise):
        add_noise = True
        noise_level = 0.5


class G1StandingCfgPPO(G1RoughCfgPPO):
    class policy(G1RoughCfgPPO.policy):
        init_noise_std = 1.0
        actor_hidden_dims = [128, 64]
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
        run_name = 'standing'
        experiment_name = 'g1_standing'
        
        save_interval = 100
        log_interval = 10
        resume = False
        load_run = -1
        checkpoint = -1