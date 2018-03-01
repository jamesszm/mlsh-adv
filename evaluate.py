"""
Setup the meta-learning task evluation over a particular algorithm.
"""

import random
from test_env import *
from policy_gradient import *

if __name__ == "__main__":
    # Sample task and train
    config = config('VanillaPolicyGradient')
    env = gym.make(config.env_name)
    model = PolicyGradient(env, config)

    # In each training task, the env should not change.
    for i in range(3):
        model.set_seed(random.randint(0, 100))
        model.initialize()
        model.train()

    # Need to measure time to convergence
    model.set_seed(random.randint(0, 100))
    model.initialize()
    model.train()

    # Then do the test.
