import tensorflow.contrib.rnn as rnn

from pg import *
from recurrent_mlsh_v4 import RecurrentMLSHV4


class RecurrentMLSHV5(RecurrentMLSHV4):
    def dynamic_rnn(self, cell, inputs):
        def rnn_forward(out, hidden, count):
            new_out, new_hidden = cell(inputs, hidden)
            new_out = tf.expand_dims(new_out, axis=1)
            new_out = tf.concat([out, new_out], axis=1)
            return [new_out, new_hidden, count + 1]

        hidden = cell.zero_state(1, tf.float32)
        out, hidden = cell(inputs, hidden)
        out = tf.expand_dims(out, axis=1)
        count = tf.constant(1)
        condition = lambda out, hidden, count: out[0][-1][-1] > 0

        raw_shape = list(cell.state_size)
        for i in range(len(raw_shape)):
            raw_shape[i] = rnn.LSTMStateTuple(hidden[i][0].get_shape(),
                                              hidden[i][1].get_shape())

        return tf.while_loop(condition, rnn_forward,
                             loop_vars=[out, hidden, count],
                             maximum_iterations=config.max_num_sub_policies - 1,
                             shape_invariants=[
                                 tf.TensorShape([1, None, out.get_shape()[-1]]),
                                 tuple(raw_shape), tf.TensorShape(None)])

    def policy_network(self, mlp_input, output_size, scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers, output_activation=None):

        num_sub_policies = config.max_num_sub_policies
        self.num_sub_policies = num_sub_policies

        self.state_embedding = mlp_input
        self.num_actions_plus_1 = self.action_dim + 1

        subpolicy_multi_cell = rnn.MultiRNNCell([self.single_cell(
            self.num_actions_plus_1, config.sub_policy_network, 'sub') for i in
                                                 range(
                                                     config.num_sub_policy_layers)],
                                                state_is_tuple=True)

        self.sub_policies, states, length = self.dynamic_rnn(
            subpolicy_multi_cell, inputs=self.state_embedding)

        self.sub_policies = self.sub_policies[:, :, :self.action_dim]

        master_multi_cell = rnn.MultiRNNCell([self.single_cell(
            num_units=config.max_num_sub_policies,
            cell_type=config.master_network, name='master') for i in
                                              range(config.num_master_layers)],
                                             state_is_tuple=True)

        concatenated = tf.concat([self.sub_policies, tf.tile(
            tf.expand_dims(self.state_embedding, axis=1), [1, length, 1])],
                                 axis=2)

        if config.freeze_sub_policy:
            concatenated = tf.stop_gradient(concatenated, name='stop')

        self.out, states = tf.nn.dynamic_rnn(cell=master_multi_cell,
                                             inputs=concatenated,
                                             sequence_length=length,
                                             dtype=tf.float32, scope='master')

        last_output = self.out[:, length - 1, :length]
        self.p = tf.Print(length, [length])

        if config.weight_average:
            self.weights = tf.nn.softmax(logits=last_output, dim=1)
        else:
            self.chosen_index = tf.argmax(last_output, axis=1)
            self.weights = tf.one_hot(indices=self.chosen_index, depth=length)

        final_policy = tf.reduce_sum(tf.expand_dims(self.weights[:, :length],
                                                    axis=2) * self.sub_policies,
            axis=1)

        if config.sub_policy_index > -1:
            final_policy = self.sub_policies[:, config.sub_policy_index, :]

        return final_policy


if __name__ == "__main__":
    env = gym.make(config.env_name)
    config = config('RecurrentMLSH-v4')
    model = RecurrentMLSHV5(env, config)
    model.run()
