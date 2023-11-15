import collections
import math
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import wandb
from hydra.utils import instantiate
from tensorflow_federated.python.core.impl.types import computation_types

from src.evaluation.simple_personalisation import evaluate_fn, build_personalize_fn
from src.evaluation.tff_overwrites.personalization_eval import build_personalization_eval
from src.training.tff_overwrites.keras_utils import from_keras_model
from src.training.train_centralised import train_model as centralised_train_model
from src.utils.auc_metric_wrapper import AUCWrapper
from src.utils.metric_logging import MetricsLogger
from src.utils.f_score_metrics import F1Score, CalibratedF1Score, FBetaScore, CalibratedFBetaScore
from src.utils.printing import pretty_print_results_dict


def train_model(model_fn, loss, train_data, val_data, test_data, client_train_sizes, client_class_weights, cfg):
    metrics_logger = MetricsLogger(f'{cfg.output_folder}/metrics.csv')
    # Do centralised training
    if cfg.dataset.prediction_target == 'target_death_within_primary_stay':
        class_weights = [0.528310715956949, 9.330578512396695]
    elif cfg.dataset.prediction_target == 'target_resurgery':
        class_weights = [0.6329658007849001, 2.3801827125790584]
    else:
        raise ValueError('prediction target not expected in setting class weights')
    train_size = np.sum(list(client_train_sizes.values()))
    centralised_train_data = train_data.create_tf_dataset_from_all_clients()
    centralised_val_data = val_data.create_tf_dataset_from_all_clients()
    centralised_test_data = test_data.create_tf_dataset_from_all_clients()
    trained_model, _ = centralised_train_model(model_fn(), loss, centralised_train_data, centralised_val_data, centralised_test_data,
                                               train_size, class_weights, cfg)

    # Load model
    # The following is necessary, because LDP makes the tensor spec have a fixed batch size, which breaks evaluation
    n_features = train_data.create_tf_dataset_from_all_clients().element_spec['x'].shape[1]

    def tff_model_fn(model=None):
        if model is None:
            model = model_fn()
        return from_keras_model(
            keras_model=model,
            loss=loss,
            input_spec=collections.OrderedDict([('x', tf.TensorSpec(shape=(None, n_features), dtype=tf.float64, name=None)),
                                                ('y', tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None))]),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                     AUCWrapper(curve='PR', name='auprc', from_logits=False),
                     AUCWrapper(curve='ROC', name='auroc', from_logits=False),
                     F1Score(name='f1', dtype=tf.float32),
                     FBetaScore(name='f2', beta=2.0, dtype=tf.float32),
                     CalibratedF1Score(name='calibrated_f1', dtype=tf.float32),
                     CalibratedFBetaScore(name='calibrated_f2', beta=2.0, dtype=tf.float32)]
        )

    per_client_eval_data = []
    for c_id in train_data.client_ids:
        per_client_eval_data.append(
            collections.OrderedDict(
                train_data=train_data.create_tf_dataset_for_client(c_id).unbatch(),
                val_data=val_data.create_tf_dataset_for_client(c_id).unbatch(),
                test_data=test_data.create_tf_dataset_for_client(c_id).unbatch(),
                context=c_id,
            )
        )
    personalisation_eval_process = build_personalization_eval(
        tff_model_fn,
        collections.OrderedDict(personalise=lambda: build_personalize_fn(
            lambda: instantiate(cfg.training.client_optimizer),
            batch_size=cfg.training.batch_size,
            num_epochs=(10 if cfg.training.personalisation else 0),
            num_epochs_per_eval=1,
            shuffle=False)),
        evaluate_fn,
        context_tff_type=computation_types.to_type(tf.string)
    )
    def perform_per_client_evaluation(model):
        per_client_metrics = personalisation_eval_process(model, per_client_eval_data)

        if 'calibrated_f1' in per_client_metrics['baseline_metrics'].keys():
            per_client_metrics['baseline_metrics'].update({
                'calibrated_f1': per_client_metrics['baseline_metrics']['calibrated_f1'][0],
                'calibrated_recall': per_client_metrics['baseline_metrics']['calibrated_f1'][1],
                'calibrated_precision': per_client_metrics['baseline_metrics']['calibrated_f1'][2],
                'calibrated_threshold': per_client_metrics['baseline_metrics']['calibrated_f1'][3]
            })
        if 'calibrated_f2' in per_client_metrics['baseline_metrics'].keys():
            per_client_metrics['baseline_metrics'].update({
                'calibrated_f2': per_client_metrics['baseline_metrics']['calibrated_f2'][0],
                'calibrated_f2_recall': per_client_metrics['baseline_metrics']['calibrated_f2'][1],
                'calibrated_f2_precision': per_client_metrics['baseline_metrics']['calibrated_f2'][2],
                'calibrated_f2_threshold': per_client_metrics['baseline_metrics']['calibrated_f2'][3]
            })
        if cfg.training.personalisation:
            for top_level_key in per_client_metrics['personalise'].keys():
                if top_level_key in ['early_stopping_round', 'num_train_examples']:
                    continue
                if 'calibrated_f1' in per_client_metrics['personalise'][top_level_key].keys():
                    per_client_metrics['personalise'][top_level_key].update({
                        'calibrated_f1': per_client_metrics['personalise'][top_level_key]['calibrated_f1'][0],
                        'calibrated_recall': per_client_metrics['personalise'][top_level_key]['calibrated_f1'][1],
                        'calibrated_precision': per_client_metrics['personalise'][top_level_key]['calibrated_f1'][2],
                        'calibrated_threshold': per_client_metrics['personalise'][top_level_key]['calibrated_f1'][3]
                    })
                if 'calibrated_f2' in per_client_metrics['personalise'][top_level_key].keys():
                    per_client_metrics['personalise'][top_level_key].update({
                        'calibrated_f2': per_client_metrics['personalise'][top_level_key]['calibrated_f2'][0],
                        'calibrated_f2_recall': per_client_metrics['personalise'][top_level_key]['calibrated_f2'][1],
                        'calibrated_f2_precision': per_client_metrics['personalise'][top_level_key]['calibrated_f2'][2],
                        'calibrated_f2_threshold': per_client_metrics['personalise'][top_level_key]['calibrated_f2'][3]
                    })
        return per_client_metrics

    from tensorflow_federated.python.learning import model_utils
    tff_model = model_utils.ModelWeights.from_model(tff_model_fn(trained_model))
    per_client_metrics = perform_per_client_evaluation(tff_model)
    client_id_order = [cid.decode() for cid in per_client_metrics['context']]
    wandb.log({f'test_single_clients/{metric}': wandb.Histogram(value)
               for metric, value in per_client_metrics['baseline_metrics'].items()}, step=(cfg.training.total_rounds + 1))
    wandb.log({f'test_single_clients/{metric}_mean': np.mean(value)
               for metric, value in per_client_metrics['baseline_metrics'].items()}, step=(cfg.training.total_rounds + 1))
    wandb.log({f'test_single_clients/{metric}_std': np.std(value)
               for metric, value in per_client_metrics['baseline_metrics'].items()}, step=(cfg.training.total_rounds + 1))
    logging.info(f'Per Client Metrics (order: {client_id_order}):')
    pretty_print_results_dict(per_client_metrics['baseline_metrics'], cfg.training.total_rounds+1, '')

    per_client_df = pd.DataFrame(columns=client_id_order)
    for metric, value in per_client_metrics['baseline_metrics'].items():
        per_client_df = pd.concat([per_client_df, pd.Series(value, index=client_id_order, name=metric).to_frame().T])

    if cfg.training.personalisation:
        logging.info(per_client_metrics)
        logging.info(f'Per Client Metrics (Personalised):')
        early_stopping_rounds = per_client_metrics['personalise']['early_stopping_round']
        # Log personalisaton process
        base_dict = per_client_metrics['personalise']
        personalised_metrics = per_client_metrics['personalise']['epoch_10']
        for personalisation_epoch in range(1, 11):
            current_epoch_metrics = base_dict[f'epoch_{personalisation_epoch}']
            for client_index in range(cfg.training.n_total_clients):
                wandb.log({f'test_personalisation/{metric}_{client_id_order[client_index]}': metric_values[client_index]
                           for metric, metric_values in current_epoch_metrics.items()}, step=cfg.training.total_rounds + 1 + personalisation_epoch)
            early_stopping_client_indices = np.argwhere(early_stopping_rounds == personalisation_epoch)
            for client_index in early_stopping_client_indices:
                for metric in personalised_metrics:
                    personalised_metrics[metric][client_index] = current_epoch_metrics[metric][client_index]
        pretty_print_results_dict(personalised_metrics, cfg.training.total_rounds + 1, '')
        wandb.log({f'test_single_clients/{metric}': wandb.Histogram(value)
                   for metric, value in personalised_metrics.items()}, step=(cfg.training.total_rounds + 2))
        wandb.log({f'test_single_clients/{metric}_mean': np.mean(value)
                   for metric, value in personalised_metrics.items()}, step=(cfg.training.total_rounds + 2))
        wandb.log({f'test_single_clients/{metric}_std': np.std(value)
                   for metric, value in personalised_metrics.items()}, step=(cfg.training.total_rounds + 2))

        for metric, value in personalised_metrics.items():
            per_client_df = pd.concat(
                [per_client_df, pd.Series(value, index=client_id_order, name=f'{metric}_personalised').to_frame().T])

    per_client_df = per_client_df.T
    per_client_df['meta_system'] = pd.to_numeric(per_client_df.index.str.slice(start=-1), errors='ignore')
    logging.info(per_client_df)
    wandb.log({'test_single_clients/all_metrics': wandb.Table(dataframe=per_client_df)})

    return None
