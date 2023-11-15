from typing import List, Dict, Tuple
from datetime import datetime
import logging
import os

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff
from omegaconf import DictConfig, OmegaConf
import wandb
from src.utils.metric_logging import MetricsLogger

from src.training import train_centralised


def train_model(model: tf.keras.Model,
                loss_object: tf.keras.losses.Loss,
                train_data: tff.simulation.datasets.ClientData,
                val_data: tff.simulation.datasets.ClientData,
                test_data: tff.simulation.datasets.ClientData,
                client_train_sizes: Dict[str, int],
                client_class_weights: Dict[str, List[int]],
                wandb_tags: List[str],
                cfg: DictConfig) -> Tuple[List[tf.keras.Model], List[Dict]]:
    initial_model_weights = model.get_weights()
    if cfg.wandb.name is None:
        cfg.wandb.group = f'local_train_{datetime.now().strftime("%d-%m_%H-%M-%S")}'
    else:
        cfg.wandb.group = cfg.wandb.name

    all_final_metrics = []

    base_output_folder = cfg.output_folder
    for client_id in train_data.client_ids:
        logging.info(f'Training on client {client_id}')
        # Reset Model
        model.reset_states()
        model.reset_metrics()
        model.set_weights(initial_model_weights)

        # Initialise WandB
        if not cfg.wandb.disabled:
            wandb.init(project=(cfg.wandb.project or f'{cfg.model.name}_{cfg.dataset.name}'), entity="bjarnepfitzner",
                       group=cfg.wandb.group, name=f'{cfg.wandb.group}_{client_id}', tags=wandb_tags, resume='allow',
                       config=OmegaConf.to_container(cfg, resolve=True), allow_val_change=True,
                       settings=wandb.Settings(start_method="fork"))
            if cfg.dp.type == 'LDP':
                noise_in_avg_grad = cfg.dp.noise_multiplier * cfg.dp.l2_norm_clip / cfg.training.batch_size
                logging.info(f'Noise std.dev. in avg grad: {noise_in_avg_grad}')
                wandb.log({'DP/noise_in_avg_grad': noise_in_avg_grad}, step=0)
                wandb.run.summary['DP/noise_in_avg_grad'] = noise_in_avg_grad

        cfg.output_folder = f'{base_output_folder}/{client_id}'
        os.makedirs(cfg.output_folder, exist_ok=True)
        trained_model, final_metrics = train_centralised.train_model(model, loss_object,
                                                                     train_data.create_tf_dataset_for_client(client_id),
                                                                     val_data.create_tf_dataset_for_client(client_id),
                                                                     test_data.create_tf_dataset_for_client(client_id),
                                                                     client_train_sizes[client_id],
                                                                     client_class_weights[client_id],
                                                                     cfg)
        model.save(f'{cfg.output_folder}/trained_model')
        all_final_metrics.append(final_metrics)

        wandb.log(final_metrics, step=cfg.training.total_rounds)

        if not cfg.wandb.disabled and wandb.run.sweep_id is None:
            wandb.finish()

    logging.info('Aggregated metrics:')
    aggregated_metrics_logger = MetricsLogger(f'{base_output_folder}/aggregated_metrics.csv')
    for metric in all_final_metrics[0].keys():
        metric_values = [final_metrics[metric] for final_metrics in all_final_metrics]
        logging.info(f'\t{metric}: {np.mean(metric_values)} +/- {np.std(metric_values)}')
        if wandb.run is not None:
            if wandb.run.sweep_id is not None:
                aggregated_metrics_logger.log({
                    f'aggregated/{metric}_mean': np.mean(metric_values),
                    f'aggregated/{metric}_std': np.std(metric_values),
                    f'all_runs/{metric}': wandb.Histogram(metric_values)
                }, step=cfg.training.total_rounds+1)
    aggregated_metrics_logger.write_metrics_to_file()

    return None, all_final_metrics
