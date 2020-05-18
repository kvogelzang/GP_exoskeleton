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


# Import the inverted pendulum as well, as this one has been hijacked to perform some simple movement analyses
register(
    id='InvertedPendulum-v2',
    entry_point='gym.envs.mujoco:InvertedPendulumEnv',
)