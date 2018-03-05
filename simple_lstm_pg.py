import tensorflow.contrib.rnn as rnn

from policy_gradient import *
from test_env import *


class LSTMPG(PolicyGradient):
    def policy_network(self,
                       mlp_input,
                       output_size,
                       scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers,
                       output_activation=None):

        if config.env_name == "Fourrooms-v1":
            state_embedding = tf.tile(
                tf.one_hot(
                    indices=tf.cast(mlp_input, dtype=tf.int32),
                    depth=self.env.nS), [1, 1, 1])
            rnn_cell = rnn.BasicLSTMCell(num_units=self.env.action_space.n)

        else:
            state_embedding = tf.tile(
                tf.expand_dims(mlp_input, axis=1), [1, 1, 1])
            rnn_cell = rnn.BasicRNNCell(
                num_units=self.env.action_space.shape[0])

        sub_policies, states = tf.nn.dynamic_rnn(
            cell=rnn_cell,
            inputs=state_embedding,
            dtype=tf.float32,
            scope='subpolicy')

        return tf.squeeze(sub_policies, axis=1)


if __name__ == "__main__":
    config = config('LSTMPG-v1')
    env = gym.make(config.env_name)
    model = LSTMPG(env, config)
    model.run()
