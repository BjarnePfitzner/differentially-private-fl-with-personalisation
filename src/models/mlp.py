import tensorflow as tf


def mlp_model(model_cfg, input_shape, seed=None):
    def get_initialiser_with_seed(seed):
        return tf.keras.initializers.get({'class_name': model_cfg.initialiser,
                                          'config': {'seed': seed}})

    def get_regulariser():
        if model_cfg.regularisation is None:
            return None
        if model_cfg.regularisation == 'l1':
            return tf.keras.regularizers.L1(model_cfg.regularisation_factor)
        if model_cfg.regularisation == 'l2':
            return tf.keras.regularizers.L2(model_cfg.regularisation_factor)

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Input(input_shape))
    if model_cfg.dropout_input > 0:
        model.add(tf.keras.layers.Dropout(model_cfg.dropout_input, seed=seed))
    for i in range(len(model_cfg.neurons)):
        model.add(tf.keras.layers.Dense(model_cfg.neurons[i],
                                        activation=tf.nn.relu,
                                        kernel_initializer=get_initialiser_with_seed(seed + i),
                                        kernel_regularizer=get_regulariser()))
        if model_cfg.dropout > 0.0 and (
                model_cfg.dropout_after == 'all_hidden' or
                (model_cfg.dropout_after == 'last_hidden_only' and i == len(model_cfg.neurons) - 1)):
            model.add(tf.keras.layers.Dropout(model_cfg.dropout, seed=seed+i))
    model.add(tf.keras.layers.Dense(2,
                                    activation=tf.nn.softmax,
                                    kernel_initializer=get_initialiser_with_seed(seed + i)))
    return model
