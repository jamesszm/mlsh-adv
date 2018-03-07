import tensorflow.contrib.rnn as rnn

from pg import *


def build_mlp(mlp_input,
              output_size,
              scope,
              size=config.baseline_layer_size,
              n_layers=config.n_layers,
              output_activation=None):
    out = mlp_input
    with tf.variable_scope(scope):
        for i in range(n_layers):
            out = layers.fully_connected(
                out, size, activation_fn=tf.nn.relu, reuse=False)

        out = layers.fully_connected(
            out, output_size, activation_fn=output_activation, reuse=False)

    return out


class VoterFCMaster(PolicyGradient):
    def policy_network(self,
                       mlp_input,
                       output_size,
                       scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers,
                       output_activation=None):

        ppsls = []
        for i in range(config.num_sub_policies):
            ppsls.append(
                self.sub_policy_network(mlp_input, output_size,
                                        scope + "/sub" + str(i)))
        return build_mlp(
            tf.concat(ppsls, axis=1), output_size, scope + "/master")

    def sub_policy_network(self, mlp_input, output_size, scope):
        return build_mlp(mlp_input, output_size, scope)


class LSTMVoterFCMaster(VoterFCMaster):
    def sub_policy_network(self, mlp_input, output_size, scope):
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
            scope=scope)

        return tf.squeeze(sub_policies, axis=1)


if __name__ == "__main__":
    env = gym.make(config.env_name)
    config = config('VoterMaster-v1')
    model = LSTMVoterFCMaster(env, config)
    model.run()
