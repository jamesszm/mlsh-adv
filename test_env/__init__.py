from gym.envs.registration import register

import gym
from test_env.envs import *

register(
    id='Fourrooms-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': '9x9',
    })

register(
    id='KeyDoor-v1',
    entry_point='test_env.envs.key_door:KeyDoor',
)
