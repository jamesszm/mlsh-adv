import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.distributions as tfd
import tensorflow.contrib.layers as layers

from config import config
from test_env import *
from utils.general import export_plot, get_logger


# noinspection PyAttributeOutsideInit
class PolicyGradient(object):
    def policy_network(mlp_input, output_size, scope, n_layers=config.n_layers,
                       size=config.layer_size, output_activation=None):
        out = mlp_input
        with tf.variable_scope(scope):
            for i in range(n_layers):
                out = layers.fully_connected(out, size,
                                             activation_fn=tf.nn.relu,
                                             reuse=False)

            out = layers.fully_connected(out, output_size,
                                         activation_fn=output_activation,
                                         reuse=False)

        return out

    def __init__(self, env, config, logger=None):

        if not os.path.exists(config.output_path):
            os.makedirs(config.output_path)

        self.config = config
        self.logger = logger
        if logger is None:
            self.logger = get_logger(config.log_path)
        self.env = env

        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = 1
        self.action_dim = self.env.action_space.n if self.discrete else \
            self.env.action_space.shape[0]

        self.lr = self.config.learning_rate

        self.build()

    def add_placeholders_op(self):
        self.observation_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                         self.observation_dim])
        if self.discrete:
            self.action_placeholder = tf.placeholder(tf.int64, shape=None)
        else:
            self.action_placeholder = tf.placeholder(tf.float32, shape=[None,
                                                                        self.action_dim])

        # Define a placeholder for advantages
        self.advantage_placeholder = tf.placeholder(tf.float32, shape=None)

    def build_policy_network_op(self, scope="policy_network"):
        if self.discrete:
            self.action_logits = self.policy_network(
                self.observation_placeholder, self.action_dim, scope=scope)
            self.sampled_action = tf.squeeze(
                tf.multinomial(self.action_logits, 1), axis=1)
            self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.action_placeholder, logits=self.action_logits)
        else:
            action_means = self.policy_network(self.observation_placeholder,
                                               self.action_dim, scope=scope)
            log_std = tf.get_variable('log_std', shape=[self.action_dim],
                                      trainable=True)
            action_std = tf.exp(log_std)
            multivariate = tfd.MultivariateNormalDiag(loc=action_means,
                                                      scale_diag=action_std)
            self.sampled_action = tf.random_normal(
                [self.action_dim]) * action_std + action_means
            self.logprob = multivariate.log_prob(self.action_placeholder)

    def add_loss_op(self):
        self.loss = -tf.reduce_mean(self.logprob * self.advantage_placeholder)

    def add_optimizer_op(self):
        self.network_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = self.network_opt.minimize(self.loss)

    def add_baseline_op(self, scope="baseline"):
        self.baseline = tf.squeeze(
            self.policy_network(self.observation_placeholder, 1, scope=scope))
        self.baseline_target_placeholder = tf.placeholder(tf.float32,
                                                          shape=None)
        self.baseline_loss = tf.losses.mean_squared_error(
            self.baseline_target_placeholder, self.baseline)
        self.baseline_opt = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.update_baseline_op = self.baseline_opt.minimize(self.baseline_loss)

    def build(self):
        self.add_placeholders_op()
        self.build_policy_network_op()
        self.add_loss_op()
        self.add_optimizer_op()

        if self.config.use_baseline:
            self.add_baseline_op()

    def initialize(self):
        self.sess = tf.Session()
        self.add_summary()
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def add_summary(self):
        self.avg_reward_placeholder = tf.placeholder(tf.float32, shape=(),
                                                     name="avg_reward")
        self.max_reward_placeholder = tf.placeholder(tf.float32, shape=(),
                                                     name="max_reward")
        self.std_reward_placeholder = tf.placeholder(tf.float32, shape=(),
                                                     name="std_reward")

        self.eval_reward_placeholder = tf.placeholder(tf.float32, shape=(),
                                                      name="eval_reward")

        tf.summary.scalar("Avg Reward", self.avg_reward_placeholder)
        tf.summary.scalar("Max Reward", self.max_reward_placeholder)
        tf.summary.scalar("Std Reward", self.std_reward_placeholder)
        tf.summary.scalar("Eval Reward", self.eval_reward_placeholder)

        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter(self.config.output_path,
                                                 self.sess.graph)

    def init_averages(self):
        self.avg_reward = 0.
        self.max_reward = 0.
        self.std_reward = 0.
        self.eval_reward = 0.

    def update_averages(self, rewards, scores_eval):
        self.avg_reward = np.mean(rewards)
        self.max_reward = np.max(rewards)
        self.std_reward = np.sqrt(np.var(rewards) / len(rewards))

        if len(scores_eval) > 0:
            self.eval_reward = scores_eval[-1]

    def record_summary(self, t):
        fd = {
            self.avg_reward_placeholder: self.avg_reward,
            self.max_reward_placeholder: self.max_reward,
            self.std_reward_placeholder: self.std_reward,
            self.eval_reward_placeholder: self.eval_reward,
        }
        summary = self.sess.run(self.merged, feed_dict=fd)
        self.file_writer.add_summary(summary, t)

    def epsilon_greedy(self, action, eps):
        if np.random.rand(1) < eps:
            return self.env.action_space.sample()
        else:
            return action

    def sample_path(self, env, num_episodes=None, eps=None):
        episode = 0
        episode_rewards = []
        paths = []
        t = 0

        while (num_episodes or t < self.config.batch_size):
            state = env.reset()
            states, actions, rewards = [], [], []
            episode_reward = 0

            for step in range(self.config.max_ep_len):
                states.append(state)
                action = self.sess.run(self.sampled_action, feed_dict={
                    self.observation_placeholder: [[states[-1]]]
                })[0]
                action = self.epsilon_greedy(action=action,
                                             eps=self.get_epsilon(t))
                state, reward, done, info = env.step(action)
                actions.append(action)
                rewards.append(reward)
                episode_reward += reward
                t += 1
                if done:
                    print('DONE!')
                if (done or step == self.config.max_ep_len - 1):
                    episode_rewards.append(episode_reward)
                    break
                if (not num_episodes) and t == self.config.batch_size:
                    break

            path = {
                "observation": np.array(states), "reward": np.array(rewards),
                "action": np.array(actions)
            }
            paths.append(path)
            episode += 1
            if num_episodes and episode >= num_episodes:
                break

        return paths, episode_rewards

    def get_returns(self, paths):
        all_returns = []
        for path in paths:
            rewards = path["reward"]
            ret = []
            T = len(rewards)
            for t in range(T - 1, -1, -1):
                r = rewards[t] + (ret[-1] * config.gamma if t < T - 1 else 0)
                ret.append(r)
            ret = ret[::-1]
            all_returns.append(ret)
        returns = np.concatenate(all_returns)

        return returns

    def calculate_advantage(self, returns, observations):
        adv = returns
        if self.config.use_baseline:
            baseline = self.sess.run(self.baseline, feed_dict={
                self.observation_placeholder: observations
            })
            adv = returns - baseline

        if self.config.normalize_advantage:
            advantage = adv
            adv = (advantage - np.mean(advantage)) / np.std(advantage)
        return adv

    def update_baseline(self, returns, observations):
        self.sess.run(self.update_baseline_op, feed_dict={
            self.observation_placeholder: observations,
            self.baseline_target_placeholder: returns
        })

    def get_epsilon(self, t):
        return max(config.min_epsilon,
                   config.max_epsilon - float(t) / config.num_batches * (
                       config.max_epsilon - config.min_epsilon))

    def train(self):
        last_record = 0

        self.init_averages()
        scores_eval = []

        for t in range(self.config.num_batches):
            print(t, self.get_epsilon(t))
            paths, total_rewards = self.sample_path(env=self.env,
                                                    eps=self.get_epsilon(t))
            scores_eval = scores_eval + total_rewards
            observations = np.expand_dims(
                np.concatenate([path["observation"] for path in paths]), axis=1)
            actions = np.concatenate([path["action"] for path in paths])
            rewards = np.concatenate([path["reward"] for path in paths])
            returns = self.get_returns(paths)
            advantages = self.calculate_advantage(returns, observations)

            if self.config.use_baseline:
                self.update_baseline(returns, observations)
            self.sess.run(self.train_op, feed_dict={
                self.observation_placeholder: observations,
                self.action_placeholder: actions,
                self.advantage_placeholder: advantages
            })

            if (t % self.config.summary_freq == 0):
                self.update_averages(total_rewards, scores_eval)
                self.record_summary(t)

            avg_reward = np.mean(total_rewards)
            sigma_reward = np.sqrt(np.var(total_rewards) / len(total_rewards))
            msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward,
                                                                 sigma_reward)
            self.logger.info(msg)

            if self.config.record and (last_record > self.config.record_freq):
                self.logger.info("Recording...")
                last_record = 0
                self.record()

        self.logger.info("- Training done.")
        export_plot(scores_eval, "Score", config.env_name,
                    self.config.plot_output)

    def evaluate(self, env=None, num_episodes=1):
        if env == None:
            env = self.env
        paths, rewards = self.sample_path(env, num_episodes)
        avg_reward = np.mean(rewards)
        sigma_reward = np.sqrt(np.var(rewards) / len(rewards))
        msg = "Average reward: {:04.2f} +/- {:04.2f}".format(avg_reward,
                                                             sigma_reward)
        self.logger.info(msg)
        return avg_reward

    def record(self):
        env = gym.make(self.config.env_name)
        env = gym.wrappers.Monitor(env, self.config.record_path,
                                   video_callable=lambda x: True, resume=True)
        self.evaluate(env, 1)

    def run(self):
        self.initialize()
        if self.config.record:
            self.record()
        self.train()
        if self.config.record:
            self.record()


if __name__ == "__main__":
    env = gym.make(config.env_name)
    model = PolicyGradient(env, config)
    model.run()
