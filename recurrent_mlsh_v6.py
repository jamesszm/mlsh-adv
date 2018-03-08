import tensorflow.contrib.rnn as rnn

from pg import *
from recurrent_mlsh_v4 import RecurrentMLSHV4


class RecurrentMLSHV5(RecurrentMLSHV4):
    def build_policy_network_op(self, scope="policy_network"):
        if self.discrete:
            with tf.variable_scope(name_or_scope='policy', reuse=False):
                self.action_logits_1 = self.policy_network(
                    self.observation_placeholder, self.action_dim, scope=scope,
                    batch_size=1)

            with tf.variable_scope(name_or_scope='policy', reuse=True):
                self.action_logits_batch = self.policy_network(
                    self.observation_placeholder, self.action_dim, scope=scope,
                    batch_size=config.batch_size)

            self.sampled_action = tf.squeeze(
                tf.multinomial(logits=self.action_logits_1, num_samples=1),
                axis=1)
            self.logprob = -tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=self.action_placeholder, logits=self.action_logits_batch)
        else:
            with tf.variable_scope(name_or_scope='policy', reuse=False):
                action_means_1 = self.policy_network(
                    self.observation_placeholder, self.action_dim, scope=scope,
                    batch_size=1)

            with tf.variable_scope(name_or_scope='policy', reuse=True):
                action_means_batch = self.policy_network(
                    self.observation_placeholder, self.action_dim, scope=scope,
                    batch_size=config.batch_size)

            log_std = tf.get_variable('log_std', shape=[self.action_dim],
                                      trainable=True)
            action_std = tf.exp(log_std)
            multivariate = tfd.MultivariateNormalDiag(loc=action_means_batch,
                                                      scale_diag=action_std)
            self.sampled_action = tf.random_normal(
                [self.action_dim]) * action_std + action_means_1
            self.logprob = multivariate.log_prob(self.action_placeholder)

    def dynamic_rnn(self, cell, inputs, batch_size):
        def rnn_forward(out, hidden, alive, count_length):
            new_out, new_hidden = cell(inputs, hidden)
            new_out = tf.expand_dims(new_out, axis=1)
            new_out = tf.concat([out, new_out], axis=1)
            alive_now = tf.cast(out[:, -1, -1] > 0, tf.int32)
            alive *= alive_now
            return [new_out, new_hidden, alive, count_length + alive_now]

        def stop(out, hidden, alive, count_length):
            return tf.reduce_sum(alive) == 0

        hidden = cell.zero_state(batch_size, tf.float32)
        out, hidden = cell(inputs, hidden)
        out = tf.expand_dims(out, axis=1)
        count_length = tf.constant(1, shape=[batch_size])

        alive = tf.constant(1, shape=[batch_size])
        condition = stop

        raw_shape = list(cell.state_size)
        for i in range(len(raw_shape)):
            raw_shape[i] = rnn.LSTMStateTuple(hidden[i][0].get_shape(),
                                              hidden[i][1].get_shape())

        return tf.while_loop(condition, rnn_forward,
                             loop_vars=[out, hidden, alive, count_length],
                             maximum_iterations=config.max_num_sub_policies - 1,
                             shape_invariants=[tf.TensorShape(
                                 [batch_size, None, out.get_shape()[-1]]),
                                 tuple(raw_shape), tf.TensorShape(batch_size),
                                 tf.TensorShape(batch_size)])

    def policy_network(self, input, output_size, scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers, output_activation=None,
                       batch_size=None):

        num_sub_policies = config.max_num_sub_policies
        self.num_sub_policies = num_sub_policies

        self.state_embedding = input
        self.num_actions_plus_1 = self.action_dim + 1

        subpolicy_multi_cell = rnn.MultiRNNCell([self.single_cell(
            self.num_actions_plus_1, config.sub_policy_network, 'sub') for i in
                                                 range(
                                                     config.num_sub_policy_layers)],
                                                state_is_tuple=True)

        self.sub_policies, states, alive, length = self.dynamic_rnn(
            subpolicy_multi_cell, inputs=self.state_embedding,
            batch_size=batch_size)

        self.sub_policies = self.sub_policies[:, :, :self.action_dim]
        # self.sub_policies => (batch size, max_length, action_dim)
        # length.shape = (batch size)

        master_multi_cell = rnn.MultiRNNCell([self.single_cell(
            num_units=config.max_num_sub_policies,
            cell_type=config.master_network, name='master') for i in
                                              range(config.num_master_layers)],
                                             state_is_tuple=True)

        max_length = tf.reduce_max(length)

        concatenated = tf.concat([self.sub_policies, tf.tile(
            tf.expand_dims(self.state_embedding, axis=1), [1, max_length, 1])],
                                 axis=2)

        # concatenated => (batch size, max_length, action_dim + state_space)

        if config.freeze_sub_policy:
            concatenated = tf.stop_gradient(concatenated, name='stop')

        self.out, states = tf.nn.dynamic_rnn(cell=master_multi_cell,
                                             inputs=concatenated,
                                             sequence_length=length,
                                             dtype=tf.float32, scope='master')

        # self.out => (batch size, max_length, max_num_sub)

        ranges = tf.expand_dims(
            tf.tile(tf.expand_dims(tf.range(max_length), axis=0),
                    [batch_size, 1]), axis=2)
        # ranges => (batch size, max_length, 1)
        mask = (tf.equal(ranges,
                         tf.expand_dims(tf.expand_dims(length - 1, axis=1),
                                        axis=2)))
        # mask => (batch size, max_length, 1)

        candidate_logits = tf.reduce_sum(self.out * tf.cast(mask, tf.float32),
                                         axis=1)
        # candidate logits => (batch size, max_num_sub)

        logit_ranges = tf.tile(
            tf.expand_dims(tf.range(config.max_num_sub_policies), axis=0),
            [batch_size, 1])
        # logit ranges => (batch size, max_num_sub)

        logit_mask = (logit_ranges < tf.expand_dims(length - 1, axis=1))
        negative_logit_mask = (
            logit_ranges >= tf.expand_dims(length - 1, axis=1))
        # negative_logit_mask, logit mask => (batch size, max_num_sub)

        mins = tf.reduce_min(candidate_logits, axis=1, keep_dims=True)
        # mins => (batch size, 1)

        reset_logits = candidate_logits * tf.cast(logit_mask, tf.float32)
        # reset_logits => (batch size, max_num_sub)

        offset = tf.cast(negative_logit_mask, tf.float32) * mins
        # offset => (batch size, max_num_sub)

        logits = reset_logits + offset

        if config.weight_average:
            self.weights = tf.nn.softmax(logits=self.last_output, dim=1)
        else:
            self.chosen_index = tf.cast(tf.argmax(logits, axis=1), tf.int32)
            tf.Assert(tf.reduce_sum(
                tf.cast(self.chosen_index < length, tf.int32)) == batch_size,
                      [self.chosen_index, length])
            self.weights = tf.one_hot(indices=self.chosen_index,
                                      depth=max_length)

        final_policy = tf.reduce_sum(
            tf.expand_dims(self.weights[:, :max_length],
                           axis=2) * self.sub_policies, axis=1)

        if config.sub_policy_index > -1:
            final_policy = self.sub_policies[:, config.sub_policy_index, :]
        print('finish building network!')
        return final_policy


if __name__ == "__main__":
    env = gym.make(config.env_name)
    config = config('RecurrentMLSH-v6')
    model = RecurrentMLSHV5(env, config)
    model.run()
