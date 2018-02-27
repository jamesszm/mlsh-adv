import tensorflow.contrib.rnn as rnn

from policy_gradient import *
from test_env import *


class RecurrentMLSH(PolicyGradient):
    def policy_network(self, mlp_input, output_size, scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers, output_activation=None):

        if config.env_name == "Fourrooms-v1":
            state_embedding = tf.tile(
                tf.one_hot(indices=tf.cast(mlp_input, dtype=tf.int32),
                           depth=self.env.nS), [1, config.num_sub_policies, 1])
            rnn_cell = rnn.BasicLSTMCell(num_units=self.env.action_space.n)

        else:
            state_embedding = tf.tile(tf.expand_dims(mlp_input, axis=1),
                                      [1, config.num_sub_policies, 1])
            rnn_cell = rnn.BasicRNNCell(
                num_units=self.env.action_space.shape[0])

        sub_policies, states = tf.nn.dynamic_rnn(cell=rnn_cell,
                                                 inputs=state_embedding,
                                                 dtype=tf.float32,
                                                 scope='subpolicy')

        lstm_cell = rnn.BasicRNNCell(num_units=config.num_sub_policies)

        # hidden_state[0] = layers.fully_connected(inputs=state_embedding,
        # num_outputs=config.num_sub_policies)
        concatenated = tf.concat([sub_policies, state_embedding], axis=2)
        out, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=concatenated,
                                        dtype=tf.float32, scope='master')
        last_output = out[:, -1, :]

        weights = tf.nn.softmax(logits=last_output, dim=1)

        final_policy = tf.reduce_sum(
            tf.expand_dims(weights, axis=2) * sub_policies, axis=1)

        return final_policy


if __name__ == "__main__":
    config = config('RecurrentMLSH-v1')
    env = gym.make(config.env_name)
    model = RecurrentMLSH(env, config)
    model.run()
