import tensorflow as tf


class config():
    # Change env_name for the different experiments
    # env_name = "CartPole-v0"
    # env_name = "InvertedPendulum-v1"
    env_name = "Fourrooms-v1"

    record = False

    batch_size_by_env = {
        "Fourrooms-v1": 200
    }

    lr_by_env = {
        "Fourrooms-v1": 3e-2
    }

    gamma_by_env = {
        "Fourrooms-v1": 1.0
    }

    # model and training config
    num_batches = 100  # number of batches trained on
    batch_size = batch_size_by_env[env_name]
    # number of steps used to compute each policy update
    max_ep_len = min(1000, batch_size)  # maximum episode length
    learning_rate = lr_by_env[env_name]
    gamma = gamma_by_env[env_name]  # the discount factor
    use_baseline = True
    normalize_advantage = True
    # parameters for the policy and baseline models
    n_layers = 4
    layer_size = 64
    activation = tf.nn.relu

    # since we start new episodes for each batch
    assert max_ep_len <= batch_size
    if max_ep_len < 0:
        max_ep_len = batch_size

    output_path = "results/%s-bs=%s-baseline=%s-lr=%s-layers=%sx%s/" % (
        env_name, batch_size, use_baseline, learning_rate, n_layers, layer_size)

    # output config

    model_output = output_path + "model.weights/"
    log_path = output_path + "log.txt"
    plot_output = output_path + "scores.png"
    record_path = output_path
    record_freq = 5
    summary_freq = 1
