import tensorflow.contrib.rnn as rnn

from pg import *
from recurrent_mlsh_v3 import RecurrentMLSHV3


class RecurrentMLSHV4(RecurrentMLSHV3):
    def policy_network(self, mlp_input, output_size, scope,
                       size=config.baseline_layer_size,
                       n_layers=config.n_layers, output_activation=None):

        proposed_num_sub_policies = layers.fully_connected(mlp_input, 1)
        num_sub_policies_per_batch = tf.cast(
            tf.minimum(proposed_num_sub_policies, config.max_num_sub_policies)[
            :, 0], tf.int32) + 1

        num_sub_policies = num_sub_policies_per_batch[0]
        self.num_sub_policies = num_sub_policies

        if str(config.env_name).startswith("Fourrooms"):
            self.state_embedding = tf.tile(
                tf.one_hot(indices=tf.cast(mlp_input, dtype=tf.int32),
                           depth=self.env.nS),
                [1, config.max_num_sub_policies, 1])
            num_actions = self.env.action_space.n

        else:
            self.state_embedding = tf.tile(tf.expand_dims(mlp_input, axis=1),
                                           [1, config.max_num_sub_policies, 1])
            num_actions = self.env.action_space.shape[0]

        self.state_embedding = self.state_embedding[:, :num_sub_policies, :]
        subpolicy_multi_cell = rnn.MultiRNNCell(
            [self.single_cell(num_actions, config.sub_policy_network) for i in
             range(config.num_sub_policy_layers)], state_is_tuple=True)

        self.sub_policies, states = tf.nn.dynamic_rnn(cell=subpolicy_multi_cell,
                                                      inputs=self.state_embedding,
                                                      sequence_length=num_sub_policies_per_batch,
                                                      dtype=tf.float32,
                                                      scope='subpolicy')

        self.sub_policies = self.sub_policies[:, :num_sub_policies, :]

        master_multi_cell = rnn.MultiRNNCell(
            [self.single_cell(config.num_sub_policies, config.master_network)
             for i in range(config.num_master_layers)], state_is_tuple=True)

        concatenated = tf.concat([self.sub_policies, self.state_embedding],
                                 axis=2)

        if config.freeze_sub_policy:
            concatenated = tf.stop_gradient(concatenated, name='stop')

        self.out, states = tf.nn.dynamic_rnn(cell=master_multi_cell,
                                             inputs=concatenated,
                                             sequence_length=num_sub_policies_per_batch,
                                             dtype=tf.float32, scope='master')

        last_output = self.out[:, num_sub_policies - 1, :num_sub_policies]

        if config.weight_average:
            self.weights = tf.nn.softmax(logits=last_output, dim=1)
        else:
            self.chosen_index = tf.argmax(last_output, axis=1)
            self.weights = tf.one_hot(indices=self.chosen_index,
                                      depth=num_sub_policies)

        final_policy = tf.reduce_sum(
            tf.expand_dims(self.weights, axis=2) * self.sub_policies, axis=1)

        if config.sub_policy_index > -1:
            final_policy = self.sub_policies[:, config.sub_policy_index, :]

        return final_policy


if __name__ == "__main__":
    env = gym.make(config.env_name)
    config = config('RecurrentMLSH-v4')
    model = RecurrentMLSHV4(env, config)
    model.run()
