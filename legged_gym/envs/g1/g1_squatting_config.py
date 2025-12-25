from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
# FIXED: Pointing to the specific G1 config location
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO

class G1SquattingCfg(G1RoughCfg):
    class init_state(G1RoughCfg.init_state):
        # Start slightly lower to be in the middle of the squat range
        pos = [0.0, 0.0, 0.75] 

    class env(G1RoughCfg.env):
        num_observations = 47
        num_privileged_obs = 50
        num_actions = 12

    class commands(G1RoughCfg.commands):
        # Zero out velocity commands so it doesn't try to walk
        class ranges:
            lin_vel_x = [0.0, 0.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]

    class rewards(G1RoughCfg.rewards):
        base_height_target = 0.75 # Mean height of the squat
        squat_amp = 0.15          # Amplitude of movement (+/- 0.15m)
        squat_freq = 0.5          # Frequency in Hz (slower is stable)

        class scales(G1RoughCfg.rewards.scales):
            # --- Disable Walking Rewards ---
            tracking_lin_vel = 0.0
            tracking_ang_vel = 0.0
            feet_air_time = 0.0
            feet_swing_height = 0.0
            
            # --- Disable Penalties that fight squatting ---
            lin_vel_z = 0.0      # We need vertical velocity!
            base_height = 0.0    # We use a dynamic target instead
            
            # --- Squatting Specific Rewards ---
            track_squat_z = 3.0  # Main task reward
            feet_stuck = 1.0     # Encourages keeping feet planted
            feet_slip = -0.5     # Penalize feet moving
            
            # --- Stability ---
            orientation = -1.0
            dof_acc = -2.5e-7
            dof_vel = -1e-3
            action_rate = -0.01

class G1SquattingCfgPPO(G1RoughCfgPPO):
    class runner(G1RoughCfgPPO.runner):
        experiment_name = 'g1_squatting'
        run_name = ''