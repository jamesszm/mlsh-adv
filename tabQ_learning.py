import numpy as np
import math
import gym
import os
import random
import time
import datetime
import matplotlib.pyplot as plt
from test_env import *

def eps_greedy_policy(Q, eps, s, nA):
  if random.random() < eps:
    # print 'random'
    return random.choice(xrange(nA))
  else:
    # print 'picking max'
    return np.argmax(Q[s], axis=0)

def get_time_stamp():
  return datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M%S')


def learn_Q_QLearning(env, num_episodes=5000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  
  ############################
  # YOUR IMPLEMENTATION HERE #
  ############################

  # Q = np.zeros((env.nS, env.nA))
  # Q = np.ones((env.nS, env.nA)) * 1.0
  Q = np.ones((env.nS, env.nA))
  eps = e
  total_score = 0
  episode_length = 0
  running_aves = []
  for episode_iter in xrange(num_episodes):
    # print 'eps = %s' % eps
    s = env.reset()
    done = False
    t = 0
    while not done:
      a = eps_greedy_policy(Q, eps, s, env.nA)
      snext, r, done, _ = env.step(a)
      total_score += r

      # if r == 1:
      #   print 'win episode_iter = %s' % episode_iter

      # update Q
      if done:
        Q[s][a] += lr * (r - Q[s][a])
      else:
        Q[s][a] += lr * (r + gamma * np.max(Q[snext], axis=0) - Q[s][a])

      # print 'episode_iter = %s, t = %s:  s = %s, a = %s, r = %s, snext = %s, done = %s, Q[%s][%s] = %s' % (episode_iter, t, s, a, r, snext, done, s, a, Q[s][a])

      s = snext

      t += 1

    episode_length += t
    eps *= decay_rate

    running_ave = total_score / (episode_iter + 1)
    running_aves.append(running_ave)

    if episode_iter % 100 == 0:
      print 'total_score = %s, # episode = %s, ave = %s, epi_length = %s, epsilon = %s' % (
        total_score, episode_iter + 1, running_ave, episode_length / (episode_iter + 1), eps)

  # entire

  plt.plot(range(1, num_episodes + 1), running_aves)
  plt.xlabel('Epoch')
  plt.ylabel('Average Reward')
  plt.xlim([0, num_episodes])
  plt.ylim([0, math.ceil(max(running_aves) * 10) / 10 * 1.1])
  # plt.savefig('results/running_aves_%s.png' % get_time_stamp())
  plt.savefig('results/fourrooms_tabq.png')

  # first 1000
  # plt.clf()
  # plt.plot(range(1, 1001), running_aves[:1000])
  # plt.xlabel('Epoch')
  # plt.ylabel('Average Reward')
  # plt.xlim([0, 1000])
  # plt.ylim([0, max(running_aves[:1000]) + 0.05])
  # # plt.savefig('running_aves_1st_1000_epsisode_%s.png' % get_time_stamp())
  # plt.savefig('results/fourrooms_tabq.png')

  return Q

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward

# Feel free to run your own debug code in main!
def main():
  if not os.path.exists('./results'):
    os.makedirs('./results')
  env = gym.make('Fourrooms-v1')
  # Q = learn_Q_QLearning(env, num_episodes=5000, e=1.0, gamma=0.9995, lr=0.2, decay_rate=0.997)
  Q = learn_Q_QLearning(env, num_episodes=400, e=1.0, decay_rate=0.92, lr=0.3)
  print('Q =\n%s' % Q)
  # render_single_Q(env, Q)

if __name__ == '__main__':
    main()
