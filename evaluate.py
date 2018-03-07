"""
Setup the meta-learning task evluation over a particular algorithm.
"""

import random
from test_env import *
from pg import *
from recurrent_mlsh_v1 import *
from simple_lstm_pg import *

if __name__ == "__main__":
    # Sample task and train
    config = config('PG-v1')
    env = gym.make(config.env_name)
    model = PolicyGradient(env, config)

    model.initialize()
    # In each training task, the env should not change.
    for i in range(3):
        model.set_seed(random.randint(0, 100))
        model.train()

    # Need to measure time to convergence
    model.set_seed(random.randint(0, 100))
    model.train()

    # Then do the test.
