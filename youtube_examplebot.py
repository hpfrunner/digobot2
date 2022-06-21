import numpy as np
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize, VecCheckNan
from stable_baselines3.ppo import MlpPolicy

from rlgym.utils.obs_builders import AdvancedObs
from rlgym.utils.state_setters import DefaultState
from rlgym.utils.terminal_conditions.common_conditions import TimeoutCondition, NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv

if __name__ == '__main__':  # Required for multiprocessing
    frame_skip = 8          # Number of ticks to repeat an action
    half_life_seconds = 5   # Easier to conceptualize, after this many seconds the reward discount is 0.5

    fps = 120 / frame_skip
    gamma = np.exp(np.log(0.5) / (fps * half_life_seconds))
    agents_per_match = 2
    num_instance = 1
    target_steps = 100_000
    steps = target_steps // (num_instance * agents_per_match)
    batch_size = steps
    
    print(f" fps={fps}, gamma={gamma})")

    def get_match():
        return Match(
            team_size=3,
            tick_skip=frame_skip,
            reward_function=VelocityPlayerToBallReward(),
            self_play=True,
            terminal_conditions=[TimeoutCondition(round(fps * 30)), GoalScoredCondition()],
            builder=AdvancedObs(),
            state_setter=DefaultState(),
            action_parser=DiscreteAction()

        )


    env = SB3MultipleInstanceEnv(get_match, 2)
    env = VecCheckNan(env)
    env = VerMonitor(env)
    env = VecNormalize(env, norm_obs=False, gamma=gamma)



    model = PPO(
        MlpPolicy,
        env,
        n_epochs
        learning_rate=5e-5,
        ent_coef=0.01,
        gamma=gamma,
        verbose=3
        batch_size=4096
        n_steps=4096
        tensorboard_log="logs",
        device="auto"

    )



    callback = CheckpointCallBack(round(1_000_000 / env.num_envs), save_path="policy"n name_prefix=" rl_model" )
