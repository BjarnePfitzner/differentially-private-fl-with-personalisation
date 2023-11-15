import math
import logging

import numpy as np
import tensorflow as tf
import tensorflow_privacy as tfp
import wandb
from hydra.utils import instantiate

from src.differential_privacy.rdp_accountant import RDPAccountant
from src.utils.auc_metric_wrapper import AUCWrapper
from src.utils.f_score_metrics import F1Score, CalibratedF1Score, FBetaScore, CalibratedFBetaScore
from src.utils.early_stopping import EarlyStopping
from src.utils.metric_logging import MetricsLogger
from src.utils.printing import pretty_print_results_dict


def train_model(model, loss_object,
                train_data: tf.data.Dataset, val_data: tf.data.Dataset, test_data: tf.data.Dataset,
                train_size, class_weights, cfg):
    metrics_logger = MetricsLogger(f'{cfg.output_folder}/metrics.csv')
    max_steps = math.inf
    if cfg.dp.type == 'LDP':
        dp_accountant = RDPAccountant(q=cfg.training.batch_size / train_size,
                                      z=cfg.dp.noise_multiplier,
                                      N=train_size,
                                      max_eps=cfg.dp.epsilon,
                                      target_delta=cfg.dp.delta,
                                      dp_type='local')
        if cfg.training.client_optimizer._target_ == 'tensorflow.keras.optimizers.SGD':
            optimizer_class = tfp.VectorizedDPKerasSGDOptimizer
        elif cfg.training.client_optimizer._target_ == 'tensorflow.keras.optimizers.Adam':
            optimizer_class = tfp.VectorizedDPKerasAdamOptimizer
        optimiser = optimizer_class(
            l2_norm_clip=float(cfg.dp.l2_norm_clip),
            noise_multiplier=float(cfg.dp.noise_multiplier),
            num_microbatches=cfg.training.batch_size,
            learning_rate=cfg.training.client_optimizer.learning_rate)
        batches_per_epoch = train_size / cfg.training.batch_size
        if cfg.dp.noise_multiplier > 0:
            max_steps, eps, _ = dp_accountant.get_maximum_n_steps(base_n_steps=batches_per_epoch, n_steps_increment=batches_per_epoch)
            max_epochs = math.ceil(max_steps / batches_per_epoch)
            metrics_logger.log({'DP/max_steps': max_steps,
                       'DP/max_epochs': max_epochs}, step=0)
            if max_steps == 0:
                logging.info('DP budget is too small to run any steps')
                return model, {'test/loss': 0.0,
                               'test/accuracy': 0.0,
                               'test/auprc': 0.0,
                               'test/auroc': 0.0,
                               'test/f1': 0.0,
                               'test/f2': 0.0,
                               'test/calibrated_f1': 0.0,
                               'test/calibrated_f2': 0.0
                               }
            # elif max_epochs > cfg.training.total_rounds:
            #     logging.info(f'DP budget is way too large, allowing {max_steps} rounds. Stopping run early - Set smaller noise multiplier')
            #     return model, {'test/loss': 0.0,
            #                    'test/accuracy': 0.0,
            #                    'test/auprc': 0.0,
            #                    'test/auroc': 0.0,
            #                    'test/f1': 0.0}
            logging.info(f'Can run for {max_steps} batches ({max_epochs} epochs) with epsilon {eps}')
    else:
        optimiser = instantiate(cfg.training.client_optimizer)

    # Metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy'),
        #'calibrated_f1': CalibratedF1Score(name='train_calibrated_f1', dtype=tf.float32),
        'f1': F1Score(name='train_f1', dtype=tf.float32),
        'f2': FBetaScore(name='train_f2', beta=2.0, dtype=tf.float32),
        'auprc': AUCWrapper(curve='PR', name='train_AUPRC', from_logits=False),
        'auroc': AUCWrapper(curve='ROC', name='train_AUROC', from_logits=False)
    }

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy'),
        #'calibrated_f1': CalibratedF1Score(name='val_calibrated_f1', dtype=tf.float32),
        'f1': F1Score(name='val_f1', dtype=tf.float32),
        'f2': FBetaScore(name='val_f2', beta=2.0, dtype=tf.float32),
        'auprc': AUCWrapper(curve='PR', name='val_AUPRC', from_logits=False),
        'auroc': AUCWrapper(curve='ROC', name='val_AUROC', from_logits=False)
    }

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_metrics = {
        'accuracy': tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy'),
        'calibrated_f1': CalibratedF1Score(name='test_calibrated_f1', dtype=tf.float32),
        'calibrated_f2': CalibratedFBetaScore(name='test_calibrated_f2', beta=2.0, dtype=tf.float32),
        'f1': F1Score(name='test_f1', dtype=tf.float32),
        'f2': FBetaScore(name='test_f2', beta=2.0, dtype=tf.float32),
        'auprc': AUCWrapper(curve='PR', name='test_AUPRC', from_logits=False),
        'auroc': AUCWrapper(curve='ROC', name='test_AUROC', from_logits=False)
    }

    all_metrics = ([train_loss, val_loss, test_loss] +
                   list(train_metrics.values()) + list(val_metrics.values()) + list(test_metrics.values()))

    @tf.function
    def train_step(x, y):
        sample_weights = None
        if cfg.training.use_sample_weights:
            sample_weights = tf.gather(class_weights, y)
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_object(y, predictions, sample_weight=sample_weights)
        grads_and_vars = optimiser._compute_gradients(loss, model.trainable_variables, tape=tape)
        optimiser.apply_gradients(grads_and_vars)

        train_loss(loss)
        for metric in train_metrics.values():
            metric(y, predictions)

    @tf.function
    def test_step(x, y, loss_metric, metrics):
        predictions = model(x, training=False)
        sample_weights = None
        if cfg.training.use_sample_weights:
            sample_weights = tf.gather(class_weights, y)
        t_loss = loss_object(y, predictions, sample_weight=sample_weights)

        loss_metric(t_loss)
        for metric in metrics.values():
            metric(y, predictions)

        return predictions

    # MAIN TRAINING LOOP
    early_stopper = EarlyStopping(cfg.early_stopping.metric_name,
                                  0,
                                  cfg.early_stopping.patience,
                                  cfg.early_stopping.mode
                                  )
    if cfg.early_stopping.metric_name == 'loss':
        early_stopper_metric = val_loss
    elif cfg.early_stopping.metric_name in val_metrics.keys():
        early_stopper_metric = val_metrics[cfg.early_stopping.metric_name]
    else:
        raise ValueError('Early Stopping metric not calculated')

    total_steps = 0
    for round in range(cfg.training.total_rounds):
        # Reset the metrics at the start of the next epoch
        for metric in all_metrics:
            metric.reset_states()

        if round % 5 == 0:
            # EVAL
            for batch in test_data:
                test_step(batch['x'], batch['y'], test_loss, test_metrics)
            calibrated_f1, calibrated_recall, calibrated_precision, threshold = test_metrics['calibrated_f1'].result()
            calibrated_f2, calibrated_f2_recall, calibrated_f2_precision, f2_threshold = test_metrics['calibrated_f2'].result()
            result_dict = {f'test/{metric_name}': metric.result().numpy() for metric_name, metric in test_metrics.items()
                           if not metric_name.startswith('calibrated')}
            result_dict.update({'test/calibrated_f1': calibrated_f1.numpy(),
                                'test/calibrated_recall': calibrated_recall.numpy(),
                                'test/calibrated_precision': calibrated_precision.numpy(),
                                'test/calibrated_threshold': threshold.numpy(),
                                'test/calibrated_f2': calibrated_f2.numpy(),
                                'test/calibrated_f2_recall': calibrated_f2_recall.numpy(),
                                'test/calibrated_f2_precision': calibrated_f2_precision.numpy(),
                                'test/calibrated_f2_threshold': f2_threshold.numpy(),
                                'test/loss': test_loss.result().numpy()})
            metrics_logger.log(result_dict, step=round)
            if round < 25 or round % 25 == 0:
                pretty_print_results_dict(result_dict, round)

        # TRAIN
        for batch in train_data:
            total_steps += 1
            if total_steps > max_steps:
                logging.info('Reached maximum possible steps for DP')
                metrics_logger.log({'early_stopping_round': round}, step=round)
                break
            train_step(batch['x'], batch['y'])
        result_dict = {f'train/{metric_name}': metric.result().numpy() for metric_name, metric in train_metrics.items()
                       if not metric_name.startswith('calibrated')}
        result_dict.update({'train/loss': train_loss.result().numpy()})
        metrics_logger.log(result_dict, step=round)
        if total_steps > max_steps:
            # needs to break here again if DP budget is exceeded
            break

        # EARLY STOPPING
        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        for batch in val_data:
            test_step(batch['x'], batch['y'], val_loss, val_metrics)
        result_dict = {f'val/{metric_name}': metric.result().numpy() for metric_name, metric in val_metrics.items()
                       if not metric_name.startswith('calibrated')}
        result_dict.update({'val/loss': val_loss.result().numpy()})
        metrics_logger.log(result_dict, step=round)
        if cfg.early_stopping.enabled:
            if early_stopper.should_stop(early_stopper_metric.result(), model.get_weights()):
                early_stopping_round = round - early_stopper.patience
                logging.info(f'Round {round} early stopping and resetting model weights to round {early_stopping_round}')
                metrics_logger.log({'early_stopping_round': early_stopping_round}, step=round)
                model.set_weights(early_stopper.get_best_model_params())
                break

    labels = []
    pred_probs = []
    for batch in test_data:
        labels.extend(batch['y'].numpy())
        predictions = test_step(batch['x'], batch['y'], test_loss, test_metrics)
        pred_probs.extend(predictions.numpy())
    calibrated_f1, calibrated_recall, calibrated_precision, threshold = test_metrics['calibrated_f1'].result()
    calibrated_f2, calibrated_f2_recall, calibrated_f2_precision, f2_threshold = test_metrics['calibrated_f2'].result()
    result_dict = {f'test/{metric_name}': metric.result().numpy() for metric_name, metric in test_metrics.items()
                   if not metric_name.startswith('calibrated')}
    result_dict.update({'test/calibrated_f1': calibrated_f1.numpy(),
                        'test/calibrated_recall': calibrated_recall.numpy(),
                        'test/calibrated_precision': calibrated_precision.numpy(),
                        'test/calibrated_threshold': threshold.numpy(),
                        'test/calibrated_f2': calibrated_f2.numpy(),
                        'test/calibrated_f2_recall': calibrated_f2_recall.numpy(),
                        'test/calibrated_f2_precision': calibrated_f2_precision.numpy(),
                        'test/calibrated_f2_threshold': f2_threshold.numpy(),
                        'test/loss': test_loss.result().numpy()})
    metrics_logger.log(result_dict, step=round)

    pred_labels = tf.cast(np.array(pred_probs)[:, 1] > threshold.numpy(), tf.int32).numpy()

    if tf.reduce_sum(pred_labels) > 0:
        # Necessary, because if no positive labels are present, pr_curve throws an error
        metrics_logger.log({
            'test/conf_mat': wandb.plot.confusion_matrix(preds=pred_labels, y_true=labels),
            'test/pr_curve': wandb.plot.pr_curve(labels, pred_probs, classes_to_plot=[1]),
            'test/roc_curve': wandb.plot.roc_curve(labels, pred_probs, classes_to_plot=[1])
        }, step=round)

    pretty_print_results_dict(result_dict, round)
    metrics_logger.write_metrics_to_file()

    return model, result_dict
