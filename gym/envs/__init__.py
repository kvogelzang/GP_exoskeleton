from gym.envs.registration import registry, register, make, spec

# Kevin
# ----------------------------------------

register(
    id='KevinHumanoid-v0',
    entry_point='gym.envs.mujoco:Kevin_HumanoidEnv',
)

register(
    id='KevinFallingHumanoid-v0',
    entry_point='gym.envs.mujoco:Kevin_FallingHumanoidEnv',
    max_episode_steps=200,
)