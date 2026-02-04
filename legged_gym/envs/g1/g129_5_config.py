from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class G1_5Cfg(LeggedRobotCfg):
    """
    PHASE 5
    """
    
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.76]
        default_joint_angles = {
            # === LEGS (Keep Phase 3) ===
            'left_hip_pitch_joint': -0.15,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.4,
            'left_ankle_pitch_joint': -0.15,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.15,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.4,
            'right_ankle_pitch_joint': -0.15,
            'right_ankle_roll_joint': 0.0,
            
            # === WAIST ===
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            # === ARMS ===
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
        
    class env(LeggedRobotCfg.env):
        num_observations = 98
        num_privileged_obs = 101
        num_actions = 29

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        static_friction = 1.0
        dynamic_friction = 1.0
        restitution = 0.

    class commands:
        curriculum = False
        num_commands = 4
        resampling_time = 100.
        heading_command = False
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.5, 1.25]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        push_robots = True
        push_interval_s = 6              # Phase 4: 8s → Very frequent
        max_push_vel_xy = 0.8            # Phase 4: 0.6 → Maximum test

    class control(LeggedRobotCfg.control):
        control_type = 'P'

        # PD within limits of XML
        stiffness = {
            'hip_pitch': 60.0,        # Limit 88 Nm -> Safe
            'hip_roll': 80.0,         # Limit 88 Nm -> REDUCED from 120 (Critical Fix)
            'hip_yaw': 40.0,          # Limit 88 Nm -> Safe
            'knee': 120.0,            # Limit 139 Nm -> Safe (Strongest Joint)

            'ankle_pitch': 40.0,      # Limit 50 Nm -> Safe
            'ankle_roll': 40.0,       # Limit 50 Nm -> Safe

            'waist_pitch': 45.0,      # Limit 50 Nm -> REDUCED from 60 (Prevents clipping)
            'waist_roll': 45.0,       # Limit 50 Nm -> REDUCED from 60
            'waist_yaw': 45.0,        # Limit 88 Nm -> Safe

            'shoulder_pitch': 14.0,
            'shoulder_roll': 14.0,
            'shoulder_yaw': 14.0,
            'elbow': 14.0,

            'wrist_roll': 14.0,       # Limit 25 Nm (Roll is stronger) -> Safe
            'wrist_pitch': 4.5,       # Limit 5 Nm -> REDUCED from 16.78 (Critical Fix)
            'wrist_yaw': 4.5,         # Limit 5 Nm -> REDUCED from 16.78
        }

        damping = {
            
            # LEGS
            'hip_pitch': 3.0,
            'hip_roll': 6.0,          # Reduced from 8.0 (matches stiffness 80)
            'hip_yaw': 2.5,
            'knee': 8.0,              # Keep high for support

            'ankle_pitch': 2.5,
            'ankle_roll': 2.5,

            'waist_pitch': 5.0,       # High damping relative to stiffness (45) for stability
            'waist_roll': 5.0,        # High damping to kill the bobblehead effect
            'waist_yaw': 3.5,

            # ARMS
            'shoulder_pitch': 1.0,
            'shoulder_roll': 1.0,
            'shoulder_yaw': 1.0,
            'elbow': 1.0,

            'wrist_roll': 1.0,
            'wrist_pitch': 0.5,       # Reduced to match stiffness 4.5
            'wrist_yaw': 0.5,         # Reduced to match stiffness 4.5
        }
        
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1_description/g1_29dof.urdf'
        name = "g1_29dof"
        foot_name = "ankle_roll"
        penalize_contacts_on = []
        terminate_after_contacts_on = ["pelvis"]
        self_collisions = 0
        flip_visual_attachments = False
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.95
        soft_torque_limit = 0.95
        base_height_target = 0.76
        max_contact_force = 300.
        
        class scales:
            # ========================================
            # PHASE 5
            # ========================================
            
            # ========================================
            # CORE (Slight increase for excellence)
            # ========================================
            alive = 15.0
            termination = -200.0
            both_feet_contact = 20.0
            single_foot_penalty = -10.0   # Allow taking a step
            
            # ========================================
            # ORIENTATION (Final polish)
            # ========================================
            orientation = -30.0          
            base_height = -15.0          
            
            # ========================================
            # STABILITY (Minimize all motion)
            # ========================================
            ang_vel_xy = -2.0           
            lin_vel_z = -2.0            
            base_lin_vel_xy = -1.0       # Allow taking a step
            base_ang_vel_z = -3.0         
            com_position = -10.0

            # ========================================
            # CONTACT (Keep Phase 4)
            # ========================================
            contact_no_vel = -5.0
            
            # ========================================
            # POSTURE (Final convergence)
            # ========================================
            dof_pos_default = -5.5        
            feet_parallel = -5.0         
            ankle_stability = 3.0     
            
            # ========================================
            # SEGMENT-SPECIFIC (Final polish)
            # ========================================
            hip_pos = -10.0                
            knee_alignment = -5.0         
            waist_position = -10.0        
            arm_position = -2.0           
            
            # ========================================
            # EFFICIENCY (Optimize energy - key for Phase 5)
            # ========================================
            dof_vel = -5e-4               
            dof_acc = -1e-7              
            action_rate = -5e-4          
            torques = -6e-5              
            
            # ========================================
            # SAFETY (Slight increase)
            # ========================================
            collision = -12.0             
            dof_pos_limits = -10.0       
            torque_limits = -2.0       
            
            # ========================================
            # DISABLED
            # ========================================
            feet_stumble = 0.0
            contact = 0.0
    
        only_positive_rewards = False
        tracking_sigma = 0.25

    class normalization:
        class obs_scales:
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
        clip_observations = 100.
        clip_actions = 10.

    class noise:
        add_noise = True
        noise_level = 0.2              # Phase 4: 0.25 → Further reduce
        
        class noise_scales:
            dof_pos = 0.005            # Phase 4: 0.006 (reduce)
            dof_vel = 0.25             # Phase 4: 0.3 (reduce)
            lin_vel = 0.05             # Phase 4: 0.06 (reduce)
            ang_vel = 0.10             # Phase 4: 0.12 (reduce)
            gravity = 0.02             # Phase 4: 0.03 (reduce)

    class viewer:
        ref_env = 0
        pos = [3, 3, 2]
        lookat = [0., 0, 1.]

    class sim:
        dt = 0.005
        substeps = 1
        gravity = [0., 0., -9.81]
        up_axis = 1

        class physx:
            num_threads = 10
            solver_type = 1
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01
            rest_offset = 0.0
            bounce_threshold_velocity = 0.5
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23
            default_buffer_size_multiplier = 5
            contact_collection = 2


class G1_5CfgPPO(LeggedRobotCfgPPO):
    """
    PPO config for PHASE 5: Final polish
    """
    seed = 1
    
    class policy:
        init_noise_std = 0.65          # Phase 4: 0.7 → Reduce further
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        activation = 'elu'
        
    class algorithm(LeggedRobotCfgPPO.algorithm):
        value_loss_coef = 1.0
        use_clipped_value_loss = True
        clip_param = 0.2
        entropy_coef = 0.005           # Phase 4: 0.006 → Minimal exploration
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1.2e-4         # Phase 4: 1.5e-4 → Fine-tuning rate
        schedule = 'adaptive'
        gamma = 0.99
        lam = 0.95
        desired_kl = 0.01
        max_grad_norm = 1.0         
        
    class runner(LeggedRobotCfgPPO.runner):
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 24
        max_iterations = 2000           # Final polish duration
        save_interval = 100
        experiment_name = 'g1_29dof_curriculum'
        run_name = 'phase5_final_polish'
        resume = True
        load_run = 'phase4_reward_tuning'  # Continue from Phase 4