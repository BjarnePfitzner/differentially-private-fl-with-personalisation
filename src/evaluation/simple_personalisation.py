# Copyright 2020, The TensorFlow Federated Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""An example of personalization strategy."""

import collections
from typing import Any, Callable, Optional, OrderedDict, Dict

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from tensorflow_federated.python.learning.framework import dataset_reduce

_OptimizerFn = Callable[[], tf.keras.optimizers.Optimizer]
_PersonalizeFn = Callable[
    [tff.learning.Model, tf.data.Dataset, tf.data.Dataset, Any],
    OrderedDict[str, tf.Tensor],
]
_EVAL_BATCH_SIZE = 32  # Batch size used when evaluating a dataset.
_SHUFFLE_BUFFER_SIZE = 1000  # Buffer size used when shuffling a dataset.


dataset_reduce_fn = dataset_reduce.build_dataset_reduce_fn(simulation_flag=True)


@tf.function
def process_auprc(true_positives, false_positives, false_negatives):
    # TP, TN, FP, FN
    num_thresholds = len(true_positives)
    dtp = (
            true_positives[: num_thresholds - 1]
            - true_positives[1:]
    )
    p = tf.math.add(true_positives, false_positives)
    dp = p[: num_thresholds - 1] - p[1:]
    prec_slope = tf.math.divide_no_nan(
        dtp, tf.maximum(dp, 0), name="prec_slope"
    )
    intercept = true_positives[1:] - tf.multiply(prec_slope, p[1:])

    safe_p_ratio = tf.where(
        tf.logical_and(p[: num_thresholds - 1] > 0, p[1:] > 0),
        tf.math.divide_no_nan(
            p[: num_thresholds - 1],
            tf.maximum(p[1:], 0),
            name="recall_relative_ratio",
        ),
        tf.ones_like(p[1:]),
    )

    pr_auc_increment = tf.math.divide_no_nan(
        prec_slope * (dtp + intercept * tf.math.log(safe_p_ratio)),
        tf.maximum(true_positives[1:] + false_negatives[1:], 0),
        name="pr_auc_increment",
    )

    return tf.reduce_sum(pr_auc_increment, name="interpolate_pr_auc")


@tf.function
def process_auroc(true_positives, true_negatives, false_positives, false_negatives):
    num_thresholds = len(true_positives)
    # Set `x` and `y` values for the curves based on `curve` config.
    recall = tf.math.divide_no_nan(true_positives,
                                   true_positives + false_negatives)
    fp_rate = tf.math.divide_no_nan(false_positives,
                                    false_positives + true_negatives)
    x = fp_rate
    y = recall

    # Find the rectangle heights based on `summation_method`.
    heights = (y[:num_thresholds - 1] + y[1:]) / 2.

    # Sum up the areas of all the rectangles.
    return tf.reduce_sum(
        tf.multiply(x[:num_thresholds - 1] - x[1:], heights),
        name="interpolate_roc_auc")


@tf.function
def process_f_beta(true_positives, false_positives, false_negatives, beta=1.0):
    precision = tf.math.divide_no_nan(
        true_positives, true_positives + false_positives
    )
    recall = tf.math.divide_no_nan(
        true_positives, true_positives + false_negatives
    )

    mul_value = precision * recall
    add_value = (tf.math.square(beta) * precision) + recall
    mean = tf.math.divide_no_nan(mul_value, add_value)
    fbeta_score = mean * (1.0 + tf.math.square(beta))

    return fbeta_score[0]


@tf.function
def process_calibrated_f_beta(true_positives, false_positives, false_negatives, beta=1.0):
    # Compute `num_thresholds` thresholds in [0, 1]
    num_thresholds = len(true_positives)
    if num_thresholds == 1:
        thresholds = [0.5]
    else:
        thresholds = [
            (i + 1) * 1.0 / (num_thresholds - 1)
            for i in range(num_thresholds - 2)
        ]
        thresholds = [0.0] + thresholds + [1.0]

    recalls = tf.math.divide_no_nan(
        true_positives,
        tf.math.add(true_positives, false_negatives),
    )
    precisions = tf.math.divide_no_nan(
        true_positives,
        tf.math.add(true_positives, false_positives),
    )

    mul_values = precisions * recalls
    add_value = (tf.math.square(beta) * precisions) + recalls
    mean = tf.math.divide_no_nan(mul_values, add_value)
    fbeta_scores = mean * (1.0 + tf.math.square(beta))

    best_idx = tf.math.argmax(fbeta_scores)
    return fbeta_scores[best_idx], recalls[best_idx], precisions[best_idx], tf.gather(thresholds, best_idx)


@tf.function
def evaluate_fn(model: tff.learning.Model,
                dataset: tf.data.Dataset) -> Dict[str, tf.Tensor]:
    """Evaluates a model on the given dataset.

    The returned metrics include those given by `model.report_local_outputs`.
    These are specified by the `loss` and `metrics` arguments when the model is
    created by `tff.learning.from_keras_model`. The returned metrics also contain
    an integer metric with name 'num_test_examples'.

    Args:
      model: A `tff.learning.Model` created by `tff.learning.from_keras_model`.
      dataset: An unbatched `tf.data.Dataset`.

    Returns:
      An `OrderedDict` of metric names to scalar `tf.Tensor`s.
    """
    # Resets the model's local variables. This is necessary because
    # `model.report_local_outputs()` aggregates the metrics from *all* previous
    # calls to `forward_pass` (which include the metrics computed in training).
    # Resetting ensures that the returned metrics are computed on test data.
    # Similar to the `reset_states` method of `tf.keras.metrics.Metric`.
    for var in model.local_variables:
        if var.initial_value is not None:
            var.assign(var.initial_value)
        else:
            var.assign(tf.zeros_like(var))

    def reduce_fn(num_examples_sum, batch):
        output = model.forward_pass(batch, training=False)
        return num_examples_sum + output.num_examples

    # Runs `reduce_fn` over the input dataset. The final metrics can be accessed
    # by `model.report_local_outputs()`.
    num_examples_sum = dataset.batch(_EVAL_BATCH_SIZE).reduce(
        initial_state=0, reduce_func=reduce_fn)
    eval_metrics = collections.OrderedDict()
    eval_metrics['num_test_examples'] = num_examples_sum
    local_outputs = model.report_local_outputs()
    # Postprocesses the metric values. This is needed because the values returned
    # by `model.report_local_outputs()` are values of the state variables in each
    # `tf.keras.metrics.Metric`. These values should be processed in the same way
    # as the `result()` method of a `tf.keras.metrics.Metric`.
    for name, metric in local_outputs.items():
        if not isinstance(metric, list):
            raise TypeError(f'The metric value returned by `report_local_outputs` is '
                            f'expected to be a list, but found an instance of '
                            f'{type(metric)}. Please check that your TFF model is '
                            'built from a keras model.')
        if len(metric) == 2:
            # The loss and accuracy metrics used in this p13n example has two values:
            # one represents `sum`, and the other represents `count`.
            eval_metrics[name] = metric[0] / metric[1]
        elif len(metric) == 1:
            eval_metrics[name] = metric[0]
        elif len(metric) == 4:
            true_positives = metric[0]
            true_negatives = metric[1]
            false_positives = metric[2]
            false_negatives = metric[3]
            # auc metric
            if name == 'auprc':
                eval_metrics[name] = process_auprc(true_positives, false_positives, false_negatives)
            elif name == 'auroc':
                eval_metrics[name] = process_auroc(true_positives, true_negatives, false_positives, false_negatives)
            elif name == 'f1':
                eval_metrics[name] = process_f_beta(true_positives, false_positives, false_negatives)
            elif name == 'calibrated_f1':
                eval_metrics[name] = process_calibrated_f_beta(true_positives, false_positives, false_negatives)
            elif name == 'f2':
                eval_metrics[name] = process_f_beta(true_positives, false_positives, false_negatives, beta=2.0)
            elif name == 'calibrated_f2':
                eval_metrics[name] = process_calibrated_f_beta(true_positives, false_positives, false_negatives, beta=2.0)
            else:
                raise ValueError(f'Metric {name} is not in  [auprc, auroc, f1, calibrated_f1].')
        else:
            raise ValueError(f'The metric value returned by `report_local_outputs` '
                             f'is expected to be a list of length 1, 2 or 4, but found '
                             f'one with length {len(metric)}.')
    return eval_metrics


def build_personalize_fn(
        optimizer_fn: _OptimizerFn,
        batch_size: int,
        num_epochs: int,
        num_epochs_per_eval: int,
        shuffle: bool = True,
) -> _PersonalizeFn:
    """Builds a `tf.function` that represents a personalization strategy.

    The returned `tf.function` represents the optimization algorithm to run on
    a client in order to personalize a given model. It takes a
    `tff.learning.models.VariableModel` (with weights already initialized to the
    desired initial model weights), an unbatched training dataset, an unbatched
    test dataset, and an optional `context` (e.g., extra datasets) as input,
    trains a personalized model on the training dataset for `num_epochs`,
    evaluates the model on the test dataset every `num_epochs_per_eval`, and
    returns the evaluation metrics. The evaluation metrics are computed by
    `evaluate_fn` (see its docstring for more details).

    This builder function only serves as an example. Customers are allowed to
    write any personalization strategy as long as it satisfies the function
    signature specified by `_PERSONALIZE_FN_TYPE`.

    Args:
      optimizer_fn: A no-argument function that returns a
        `tf.keras.optimizers.Optimizer`.
      batch_size: An `int` specifying the batch size used in training.
      num_epochs: An `int` specifying the number of epochs used in training a
        personalized model.
      num_epochs_per_eval: An `int` specifying the frequency that a personalized
        model gets evaluated during the process of training.
      shuffle: A `bool` specifying whether to shuffle train data in every epoch.

    Returns:
      A `tf.function` that trains a personalized model, evaluates the model every
      `num_epochs_per_eval` epochs, and returns the evaluation metrics.
    """
    # Create the `optimizer` here instead of inside the `tf.function` below,
    # because a `tf.function` generally does not allow creating new variables.
    optimizer = optimizer_fn()

    @tf.function
    def personalize_fn(
            model: tff.learning.Model,
            train_data: tf.data.Dataset,
            val_data: tf.data.Dataset,
            test_data: tf.data.Dataset,
            context: Optional[Any] = None,
    ) -> OrderedDict[str, Any]:
        """A personalization strategy that trains a model and returns the metrics.

        Args:
          model: A `tff.learning.models.VariableModel`.
          train_data: An unbatched `tf.data.Dataset` used for training.
          test_data: An unbatched `tf.data.Dataset` used for evaluation.
          context: An optional object (e.g., extra dataset) used in personalization.

        Returns:
          An `OrderedDict` that maps metric names to `tf.Tensor`s or structures of
          `tf.Tensor`s containing the training and evaluation metrics.
        """
        def train_one_batch(num_examples_sum, batch):
            """Run gradient descent on a batch."""
            with tf.GradientTape() as tape:
                output = model.forward_pass(batch)
            grads = tape.gradient(output.loss, model.trainable_variables)
            optimizer.apply_gradients(
                zip(
                    tf.nest.flatten(grads), tf.nest.flatten(model.trainable_variables)
                )
            )
            # Update the number of examples.
            return num_examples_sum + output.num_examples

        # Start training.
        training_state = 0  # Number of total examples used in training.
        metrics_dict = collections.OrderedDict()
        best_loss = np.Inf
        wait = 0
        best_model_weights = model.weights
        metrics_dict['early_stopping_round'] = num_epochs
        patience = 3

        for epoch_idx in range(1, num_epochs + 1):
            perform_eval_this_round = False
            if shuffle:
                train_data = train_data.shuffle(_SHUFFLE_BUFFER_SIZE)
            training_state = train_data.batch(batch_size).reduce(
                initial_state=training_state, reduce_func=train_one_batch
            )
            val_metrics = evaluate_fn(model, val_data)
            if val_metrics['loss'] <= best_loss:
                best_loss = val_metrics['loss']
                wait = 0
                best_model_weights = model.weights
            else:
                wait += 1
                best_model_weights = best_model_weights
                if wait >= patience:
                    best_model_weights.assign_weights_to(model)
                    perform_eval_this_round = True
                    metrics_dict['early_stopping_round'] = epoch_idx - patience
                    wait = 0
                    patience = num_epochs
            # Evaluate the trained model every `num_epochs_per_eval` epochs.
            if (epoch_idx % num_epochs_per_eval == 0) or (epoch_idx == num_epochs) or perform_eval_this_round:
                metrics_dict[f'epoch_{epoch_idx}'] = evaluate_fn(model, test_data)

        # Save the training statistics.
        metrics_dict['num_train_examples'] = training_state

        return metrics_dict

    return personalize_fn
