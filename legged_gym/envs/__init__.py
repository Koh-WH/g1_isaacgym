from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from legged_gym.envs.go2.go2_config import GO2RoughCfg, GO2RoughCfgPPO
from legged_gym.envs.h1.h1_config import H1RoughCfg, H1RoughCfgPPO
from legged_gym.envs.h1.h1_env import H1Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2RoughCfg, H1_2RoughCfgPPO
from legged_gym.envs.h1_2.h1_2_env import H1_2Robot
from legged_gym.envs.g1.g1_config import G1RoughCfg, G1RoughCfgPPO
from legged_gym.envs.g1.g1_env import G1Robot
from .base.legged_robot import LeggedRobot

from legged_gym.utils.task_registry import task_registry

task_registry.register( "go2", LeggedRobot, GO2RoughCfg(), GO2RoughCfgPPO())
task_registry.register( "h1", H1Robot, H1RoughCfg(), H1RoughCfgPPO())
task_registry.register( "h1_2", H1_2Robot, H1_2RoughCfg(), H1_2RoughCfgPPO())
task_registry.register( "g1", G1Robot, G1RoughCfg(), G1RoughCfgPPO())

# Added
from legged_gym.envs.g1.g1_standing_config import G1StandingCfg, G1StandingCfgPPO
from legged_gym.envs.g1.g1_standing_env import G1StandingEnv
task_registry.register( "g1_standing", G1StandingEnv, G1StandingCfg(), G1StandingCfgPPO())
from legged_gym.envs.g1.g1_squatting_config import G1SquattingCfg, G1SquattingCfgPPO
from legged_gym.envs.g1.g1_squatting_env import G1SquattingEnv
task_registry.register("g1_squatting", G1SquattingEnv, G1SquattingCfg, G1SquattingCfgPPO)

from legged_gym.envs.g1.g129_1_config import G1_1Cfg, G1_1CfgPPO
from legged_gym.envs.g1.g129_1_env import G1_1Env
task_registry.register( "g1_1", G1_1Env, G1_1Cfg(), G1_1CfgPPO())
from legged_gym.envs.g1.g129_2_config import G1_2Cfg, G1_2CfgPPO
from legged_gym.envs.g1.g129_1_env import G1_1Env
task_registry.register( "g1_2", G1_1Env, G1_2Cfg(), G1_2CfgPPO())
from legged_gym.envs.g1.g129_3_config import G1_3Cfg, G1_3CfgPPO
from legged_gym.envs.g1.g129_1_env import G1_1Env
task_registry.register( "g1_3", G1_1Env, G1_3Cfg(), G1_3CfgPPO())
from legged_gym.envs.g1.g129_4_config import G1_4Cfg, G1_4CfgPPO
from legged_gym.envs.g1.g129_1_env import G1_1Env
task_registry.register( "g1_4", G1_1Env, G1_4Cfg(), G1_4CfgPPO())
from legged_gym.envs.g1.g129_5_config import G1_5Cfg, G1_5CfgPPO
from legged_gym.envs.g1.g129_1_env import G1_1Env
task_registry.register( "g1_5", G1_1Env, G1_5Cfg(), G1_5CfgPPO())