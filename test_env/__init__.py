from gym.envs.registration import register

from test_env.envs import *

register(
    id='Fourrooms-v1',
    entry_point='test_env.envs.fourrooms:Fourrooms',
    kwargs={
        'map_name': '9x9',
    })

register(id='Fourrooms-random-start-state-v1',
         entry_point='test_env.envs.fourrooms_random_start_state:Fourrooms',
         kwargs={
             'map_name': '9x9',
         })

register(id='Fourrooms-fixed-start-state-v1',
         entry_point='test_env.envs.fourrooms_fixed_start_state:Fourrooms',
         kwargs={
             'map_name': '9x9',
         })

register(id='KeyDoor-v1', entry_point='test_env.envs.key_door:KeyDoor', )
