from gymnasium.envs.registration import register
#from fracturesurgery import fracturesurgery_env

register(
    id='anklesurg-v0',
    entry_point='gym_fracture.envs:fracturesurgery_env'
)