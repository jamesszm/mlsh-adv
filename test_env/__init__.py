from gym.envs.registration import register
import gym
from test_env.envs import *

register(
    id='Fourrooms-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
)

register(
    id='KeyDoor-v1',
    entry_point='test_env.envs.key_door:KeyDoor',
)
