"""
Setup the meta-learning task evluation over a particular algorithm.
"""

import random
from test_env import *
from policy_gradient import *
from recurrent_mlsh_v1 import *

if __name__ == "__main__":
    # Sample task and train
    config = config('RecurrentMLSH-v1')
    env = gym.make(config.env_name)
    model = RecurrentMLSH(env, config)

    model.initialize()
    # In each training task, the env should not change.
    for i in range(3):
        model.set_seed(random.randint(0, 100))
        model.train()

    # Need to measure time to convergence
    model.set_seed(random.randint(0, 100))
    model.train()

    # Then do the test.
