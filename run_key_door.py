from test_env import *


def train(env):
    """Show an example of gym
        Parameters
                ----------
                env: gym.core.Environment
                        Environment to play on. Must have nS, nA, and P as
                        attributes.
        """
    env.seed(0)
    from gym.spaces import prng
    prng.seed(10)  # for print the location

    for eps in range(100):
        # Generate the episode
        ob = env.reset()

        for t in range(1000000):
            env.render()
            a = env.action_space.sample()
            ob, rew, done, _ = env.step(a)
            if done:
                break
        assert done
        env.render()


if __name__ == "__main__":
    env = gym.make("Fourrooms-v1")
    train(env)
