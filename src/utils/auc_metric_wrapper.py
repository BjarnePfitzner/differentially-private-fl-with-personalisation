import tensorflow as tf


class AUCWrapper(tf.keras.metrics.AUC):
    def update_state(self, y_true, y_pred, sample_weight=None):
        super().update_state(y_true, y_pred[:,  1])
