import tensorflow as tf


class config():
    def __init__(self, algorithm=None):
        if not algorithm:
            raise Exception()
        self.algorithm = algorithm

        output_path = "results/%s-bs=%s-algorithm-%s-usebaseline=%s-lr=%s" \
                      "-baselinelayers=%sx%s-numsubpolicies-%s-maxeps-%s-min" \
                      "eps-%s/" % (
                          self.env_name, self.batch_size, self.algorithm,
                          self.use_baseline, self.learning_rate, self.n_layers,
                          self.baseline_layer_size, self.num_sub_policies,
                          self.max_epsilon, self.min_epsilon)

        self.model_output = output_path + "model.weights/"
        self.log_path = output_path + "log.txt"
        self.plot_output = output_path + "scores.png"
        self.record_path = output_path
        self.output_path = output_path

    # Change env_name for the different experiments
    # env_name = "CartPole-v0"
    # env_name = "InvertedPendulum-v1"
    # env_name = "Fourrooms-v1"
    env_name = "HalfCheetah-v1"

    record = True

    batch_size_by_env = {"Fourrooms-v1": 1000, "HalfCheetah-v1": 10000}

    lr_by_env = {"Fourrooms-v1": 3e-2, "HalfCheetah-v1": 2.8e-2}

    gamma_by_env = {"Fourrooms-v1": 1.0, "HalfCheetah-v1": 0.9}
    max_epsilon = 0
    min_epsilon = 0
    # model and training config
    num_batches = 1000  # number of batches trained on
    batch_size = batch_size_by_env[env_name]
    # number of steps used to compute each policy update
    max_ep_len = min(10000, batch_size)  # maximum episode length
    learning_rate = lr_by_env[env_name]
    gamma = gamma_by_env[env_name]  # the discount factor
    use_baseline = True
    normalize_advantage = True
    # parameters for the policy and baseline models
    n_layers = 1
    baseline_layer_size = 16
    num_sub_policies = 4
    activation = tf.nn.relu

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0:
        max_ep_len = batch_size

    record_freq = 5
    summary_freq = 1
