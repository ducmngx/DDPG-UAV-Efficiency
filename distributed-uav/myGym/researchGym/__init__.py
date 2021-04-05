from gym.envs.registration import register

register(
    id='uav-v0',
    entry_point='researchGym.envs:UAVEnv',
)
