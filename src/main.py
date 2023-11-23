import os
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import traceback
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
import numpy as np
import tensorflow as tf
physical_devices = tf.config.list_physical_devices('GPU')
for device in physical_devices:
   tf.config.experimental.set_memory_growth(device, True)
import tensorflow_federated as tff

from src.data import load_dataset, print_class_imbalance
from src.data.prepare_datasets import prepare_centralised_ds, prepare_federated_ds
from src.models import get_model
from src.utils.shap import plot_shap_values


def setup_wandb(cfg: DictConfig):
    if cfg.wandb.disabled:
        logging.info('Disabled Wandb logging')
        wandb.init(mode='disabled')
        wandb_tags = None
    else:
        # Initialise WandB
        wandb_tags = [cfg.training.type, cfg.dataset.name, cfg.dataset.prediction_target]
        if cfg.dp.type != 'disabled':
            wandb_tags.append(f'{cfg.dp.type} DP')
        if cfg.training.type != 'local':
            # Local training initialises wandb for each client
            wandb.init(project=(cfg.wandb.project or f'{cfg.model.name}_{cfg.dataset.name}'), entity=cfg.wandb.entity,
                       group=cfg.wandb.group, name=cfg.wandb.name, tags=wandb_tags, resume='allow',
                       config=OmegaConf.to_container(cfg, resolve=True), allow_val_change=True,
                       settings=wandb.Settings(start_method="fork"))

    return wandb_tags


@hydra.main(config_path="../config", config_name="config")
def main(cfg: DictConfig):
    # set random seeds
    if cfg.zeed is not None:
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        random.seed(42 + cfg.zeed)
        np.random.seed(42 + cfg.zeed)
        tf.random.set_seed(42 + cfg.zeed)

    tff.backends.native.set_local_execution_context(clients_per_thread=1,
                                                    client_tf_devices=tf.config.list_logical_devices('GPU'))

    if cfg.debug:
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    else:
        logging.basicConfig(format='%(message)s', level=logging.INFO)

    # ========== Custom calculation of config values ===========
    if cfg.dp.get('maximum_noise_in_average_gradient') is not None:
        cfg.dp.noise_multiplier = round(cfg.dp.maximum_noise_in_average_gradient *
                                        cfg.training.batch_size / cfg.dp.l2_norm_clip, 2)
        logging.info(f'Setting noise multiplier to {cfg.dp.noise_multiplier} to keep maximum noise level')

    if cfg.dp.get('maximum_noise_in_average_model_update') is not None:
        cfg.dp.noise_multiplier = round(cfg.dp.maximum_noise_in_average_model_update *
                                        (cfg.training.client_sampling_prob * cfg.training.n_total_clients) / cfg.dp.l2_norm_clip, 2)
        logging.info(f'Setting noise multiplier to {cfg.dp.noise_multiplier} to keep maximum noise level')

    # WandB Setup
    wandb_tags = setup_wandb(cfg)

    # ========== Save config file ==========
    logging.info(OmegaConf.to_yaml(cfg))

    X, Y = load_dataset(cfg.dataset, split_col=cfg.training.get('split_clients_by'))
    print_class_imbalance(Y, [cfg.dataset.prediction_target])
    y = Y[cfg.dataset.prediction_target]

    # Setup model fn and loss
    model = get_model(cfg.model, (len(X.columns),), cfg.zeed)
    model.summary(print_fn=logging.info)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False,
        reduction=(tf.losses.Reduction.NONE if cfg.dp.type == 'LDP' else tf.losses.Reduction.AUTO))

    # create output folder
    os.makedirs(cfg.output_folder, exist_ok=True)

    if cfg.training.type == 'centralised':
        train_data, val_data, test_data, train_size, class_weights = prepare_centralised_ds(X, y, cfg)
        logging.info(f'Sample weights: {class_weights}')

        from src.training.train_centralised import train_model
        trained_model, _ = train_model(model, loss, train_data, val_data, test_data, train_size, class_weights, cfg)
    else:
        train_data, val_data, test_data, client_train_sizes, client_class_weights, n_input_features = (
            prepare_federated_ds(X, y, cfg))
        if cfg.training.type == 'local':
            from src.training.train_local import train_model
            trained_model = train_model(model, loss, train_data, val_data, test_data, client_train_sizes,
                                        client_class_weights, wandb_tags, cfg)
        elif cfg.training.type == 'federated':
            from src.training.train_federated import train_model
            trained_model = train_model(lambda: get_model(cfg.model, (n_input_features,), cfg.zeed), loss,
                                        train_data, val_data, test_data, client_train_sizes, client_class_weights, cfg)
        elif cfg.training.type == 'centralised_per_client_eval':
            from src.training.train_centralised_eval_federated import train_model
            trained_model = train_model(lambda: get_model(cfg.model, (n_input_features,), cfg.zeed), loss,
                                        train_data, val_data, test_data, client_train_sizes, client_class_weights, cfg)
        # Creating centralised datasets for SHAP analysis
        train_data = train_data.create_tf_dataset_from_all_clients()
        test_data = test_data.create_tf_dataset_from_all_clients()

    # Evaluate model
    if cfg.evaluation.shap and trained_model is not None:
        train_x = np.array(list(train_data.map(lambda elem: elem['x']).unbatch().as_numpy_iterator()))
        test_x = np.array(list(test_data.map(lambda elem: elem['x']).unbatch().as_numpy_iterator()))
        try:
            plot_shap_values(trained_model, test_x, train_x, X.columns)
        except AssertionError as e:
            logging.info('Encountered assertion error in SHAP calculation. Not saving SHAP plots.')
            logging.info(e)

    # Save model
    trained_model.save(f'{cfg.output_folder}/trained_model')
    logging.info('Finished training successfully.')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        logging.info(traceback.print_exc())
        exit(1)
