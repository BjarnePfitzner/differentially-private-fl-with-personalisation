import collections
import math
import logging

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_federated as tff
import tensorflow_privacy as tfp
import wandb
from hydra.utils import instantiate
from tensorflow_federated.python.core.impl.types import computation_types

from src.differential_privacy.rdp_accountant import RDPAccountant
from src.evaluation.simple_personalisation import evaluate_fn, build_personalize_fn
from src.evaluation.tff_overwrites.federated_evaluation import build_federated_evaluation
from src.evaluation.tff_overwrites.personalization_eval import build_personalization_eval
from src.training.tff_overwrites.federated_averaging import build_federated_averaging_process
from src.training.tff_overwrites.keras_utils import from_keras_model
from src.training.tff_overwrites.optimizer_utils import ClientState
from src.utils.auc_metric_wrapper import AUCWrapper
from src.utils.f_score_metrics import F1Score, CalibratedF1Score, FBetaScore, CalibratedFBetaScore
from src.utils.early_stopping import EarlyStopping
from src.utils.metric_logging import MetricsLogger
from src.utils.printing import pretty_print_results_dict


# def setup_ldp_training():

#    return learning_process, accountant

def train_model(model_fn, loss, train_data, val_data, test_data, client_train_sizes, client_class_weights, cfg):
    metrics_logger = MetricsLogger(f'{cfg.output_folder}/metrics.csv')
    # n_clients_per_round takes precedence over client_sampling_prob
    if cfg.training.n_clients_per_round is not None:
        cfg.training.client_sampling_prob = cfg.training.n_clients_per_round / cfg.training.n_total_clients
    elif cfg.training.client_sampling_prob is not None:
        cfg.training.n_clients_per_round = cfg.training.client_sampling_prob * cfg.training.n_total_clients

    if cfg.dp.type == 'LDP':
        noise_in_avg_grad = cfg.dp.noise_multiplier * cfg.dp.l2_norm_clip / cfg.training.batch_size
        logging.info(f'Noise std.dev. in avg grad: {noise_in_avg_grad}')
        metrics_logger.log({'DP/noise_in_avg_grad': noise_in_avg_grad}, step=0)
        wandb.run.summary['DP/noise_in_avg_grad'] = noise_in_avg_grad
    elif cfg.dp.type == 'CDP':
        noise_in_avg_round = cfg.dp.noise_multiplier * cfg.dp.l2_norm_clip / cfg.training.n_clients_per_round
        logging.info(f'Noise std.dev. in avg round: {noise_in_avg_round}')
        metrics_logger.log({'DP/noise_in_avg_round': noise_in_avg_round}, step=0)
        wandb.run.summary['DP/noise_in_avg_round'] = noise_in_avg_round
    # Load model
    # The following is necessary, because LDP makes the tensor spec have a fixed batch size, which breaks evaluation
    n_features = train_data.create_tf_dataset_from_all_clients().element_spec['x'].shape[1]

    def tff_model_fn():
        model = model_fn()
        return from_keras_model(
            keras_model=model,
            loss=loss,
            input_spec=collections.OrderedDict(
                [('x', tf.TensorSpec(shape=(None, n_features), dtype=tf.float64, name=None)),
                 ('y', tf.TensorSpec(shape=(None,), dtype=tf.int64, name=None))]),
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
                     AUCWrapper(curve='PR', name='auprc', from_logits=False),
                     AUCWrapper(curve='ROC', name='auroc', from_logits=False),
                     F1Score(name='f1', dtype=tf.float32),
                     FBetaScore(name='f2', beta=2.0, dtype=tf.float32),
                     CalibratedF1Score(name='calibrated_f1', dtype=tf.float32),
                     CalibratedFBetaScore(name='calibrated_f2', beta=2.0, dtype=tf.float32)]
        )

    if cfg.training.use_sample_weights:
        client_states = {cid: ClientState(client_class_weights[cid])
                         for cid in train_data.client_ids}
    else:
        client_states = {cid: ClientState([1.0, 1.0])
                         for cid in train_data.client_ids}

    # Setup aggregation factory and federated averaging process
    if cfg.dp.type == 'disabled':
        learning_process = build_federated_averaging_process(
            tff_model_fn,
            client_optimizer_fn=lambda: instantiate(cfg.training.client_optimizer),
            server_optimizer_fn=lambda: instantiate(cfg.training.server_optimizer),
            use_experimental_simulation_loop=True
        )
        accountant = None
    elif cfg.dp.type == 'LDP':
        if cfg.training.client_optimizer._target_ == 'tensorflow.keras.optimizers.SGD':
            optimizer_class = tfp.VectorizedDPKerasSGDOptimizer
        elif cfg.training.client_optimizer._target_ == 'tensorflow.keras.optimizers.Adam':
            optimizer_class = tfp.VectorizedDPKerasAdamOptimizer
        learning_process = build_federated_averaging_process(
            tff_model_fn,
            client_optimizer_fn=lambda: optimizer_class(
                l2_norm_clip=float(cfg.dp.l2_norm_clip),
                noise_multiplier=float(cfg.dp.noise_multiplier),
                num_microbatches=cfg.training.batch_size,
                learning_rate=cfg.training.client_optimizer.learning_rate),
            server_optimizer_fn=lambda: instantiate(cfg.training.server_optimizer),
            use_experimental_simulation_loop=True
        )

        # Set up Accountants
        if cfg.dp.noise_multiplier == 0:
            logging.info('Not evaluating accountants due to noise multiplier being 0')
            accountant = None
        else:
            accountant = {}
            max_local_train_steps = {}
            possible_selections = []
            for client_id, client_data_size in client_train_sizes.items():
                accountant[client_id] = RDPAccountant(q=cfg.training.batch_size / client_data_size,
                                                      z=cfg.dp.noise_multiplier,
                                                      N=client_data_size,
                                                      max_eps=cfg.dp.epsilon,
                                                      target_delta=cfg.dp.delta,
                                                      dp_type='local')
                steps_per_selection = client_data_size * cfg.training.local_epochs // cfg.training.batch_size
                max_steps, _, _ = accountant[client_id].get_maximum_n_steps(base_n_steps=100,
                                                                            n_steps_increment=steps_per_selection // cfg.training.local_epochs)
                max_local_train_steps[client_id] = max_steps
                possible_selections.append(math.ceil(max_steps / steps_per_selection))
            if all([max_steps == 0 for max_steps in max_local_train_steps.values()]):
                logging.info('Privacy requirements too tight - no rounds possible.')
                metrics_logger.log({'test/auprc': 0.0,
                                    'test/auroc': 0.0,
                                    'test/accuracy': 0.0,
                                    'test/f1': 0.0,
                                    'test/f2': 0.0,
                                    'test/calibrated_f1': 0.0,
                                    'test/calibrated_f2': 0.0
                                    }, step=0)
                return

            logging.info(
                f'{list(max_local_train_steps.values())} local steps possible ({possible_selections} selections)')
            local_train_step_budget = max_local_train_steps.copy()
            metrics_logger.log({'DP/max_steps': max(max_local_train_steps.values()),
                                'DP/max_epochs': max(possible_selections)}, step=0)

    elif cfg.dp.type == 'CDP':
        # Using the `dp_aggregator` here turns on differential privacy with adaptive
        # clipping.
        if cfg.dp.adaptive:
            aggregation_factory = tff.learning.model_update_aggregator.dp_aggregator(
                noise_multiplier=float(cfg.dp.noise_multiplier),
                clients_per_round=cfg.training.n_clients_per_round,
                zeroing=False)
        else:
            aggregation_factory = tff.aggregators.DifferentiallyPrivateFactory.gaussian_fixed(
                clients_per_round=cfg.training.n_clients_per_round,
                noise_multiplier=float(cfg.dp.noise_multiplier),
                clip=float(cfg.dp.l2_norm_clip))

        # Build a federated averaging process.
        # Typically, a non-adaptive server optimizer is used because the noise in the
        # updates can cause the second moment accumulators to become very large
        # prematurely.
        learning_process = build_federated_averaging_process(
            tff_model_fn,
            client_optimizer_fn=lambda: instantiate(cfg.training.client_optimizer),
            server_optimizer_fn=lambda: instantiate(cfg.training.server_optimizer),
            model_update_aggregation_factory=aggregation_factory,
            use_experimental_simulation_loop=True
        )

        if cfg.dp.noise_multiplier < 1e-4:
            logging.info('WARNING - Not running DP Accountant since likely a clipping test is performed')
            accountant = None
        else:
            accountant = RDPAccountant(q=cfg.training.client_sampling_prob,
                                       z=cfg.dp.noise_multiplier,
                                       N=cfg.training.n_total_clients,
                                       max_eps=cfg.dp.epsilon,
                                       target_delta=cfg.dp.delta)
            max_rounds, max_epsilon, max_order = accountant.get_maximum_n_steps(n_steps_increment=25)
            cfg.training.total_rounds = max_rounds
            if max_rounds == 0:
                logging.info('Privacy requirements too tight - no rounds possible.')
                metrics_logger.log({'test/auprc': 0.0,
                                    'test/auroc': 0.0,
                                    'test/accuracy': 0.0,
                                    'test/f1': 0.0,
                                    'test/f2': 0.0,
                                    'test/calibrated_f1': 0.0,
                                    'test/calibrated_f2': 0.0
                                    }, step=0)
                return

            logging.info(f'{max_rounds} rounds possible with eps = {max_epsilon}')
            metrics_logger.log({'DP/max_epochs': max_rounds}, step=0)
    else:
        raise ValueError(f'Unknown DP type: {cfg.dp.type}')

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

    eval_process = build_federated_evaluation(tff_model_fn, use_experimental_simulation_loop=True)
    val_data_for_tff = [val_data.create_tf_dataset_for_client(c_id) for c_id in val_data.client_ids]
    test_data_for_tff = [test_data.create_tf_dataset_for_client(c_id) for c_id in test_data.client_ids]

    def perform_evaluation(model, data_for_eval):
        metrics = eval_process(model, data_for_eval)

        if 'calibrated_f1' in metrics.keys():
            metrics.update({
                'calibrated_f1': metrics['calibrated_f1'][0],
                'calibrated_recall': metrics['calibrated_f1'][1],
                'calibrated_precision': metrics['calibrated_f1'][2],
                'calibrated_threshold': metrics['calibrated_f1'][3]
            })
        if 'calibrated_f2' in metrics.keys():
            metrics.update({
                'calibrated_f2': metrics['calibrated_f2'][0],
                'calibrated_f2_recall': metrics['calibrated_f2'][1],
                'calibrated_f2_precision': metrics['calibrated_f2'][2],
                'calibrated_f2_threshold': metrics['calibrated_f2'][3]
            })
        return metrics

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

    # Training loop.
    early_stopper = EarlyStopping(cfg.early_stopping.metric_name,
                                  0,
                                  cfg.early_stopping.patience,
                                  cfg.early_stopping.mode
                                  )
    state = learning_process.initialize()
    active_clients_list = train_data.client_ids[:]
    for round_num in range(cfg.training.total_rounds):
        if round_num % 5 == 0:
            model = state.model
            metrics = perform_evaluation(model, test_data_for_tff)

            # Logging
            metrics_logger.log({f'test/{metric}': value for metric, value in metrics.items() if
                                type(value) != collections.OrderedDict}, step=round_num)
            if round_num < 25 or round_num % 25 == 0:
                pretty_print_results_dict(metrics, round_num, '')

        # Sample client data and states
        x = np.random.uniform(size=len(active_clients_list))
        sampled_clients = [
            active_clients_list[i] for i in range(len(active_clients_list))
            if x[i] < cfg.training.client_sampling_prob]
        if len(sampled_clients) == 0:
            sampled_clients = [np.random.choice(active_clients_list)]
        sampled_train_data = [
            train_data.create_tf_dataset_for_client(client)
            for client in sampled_clients]
        sampled_client_states = [
            client_states[client]
            for client in sampled_clients]

        # check for exceeding of client privacy budget for LDP
        if cfg.dp.type == 'LDP' and cfg.dp.noise_multiplier > 0:
            metrics_logger.log({'DP/n_active_clients': len(active_clients_list)}, step=round_num)
            for i, client_id in enumerate(sampled_clients):
                steps_per_local_optimisation = client_train_sizes[
                                                   client_id] * cfg.training.local_epochs // cfg.training.batch_size
                if local_train_step_budget[client_id] > steps_per_local_optimisation:
                    local_train_step_budget[client_id] -= steps_per_local_optimisation
                else:
                    sampled_train_data[i] = sampled_train_data[i].take(local_train_step_budget[client_id])
                    active_clients_list.remove(client_id)
                    logging.info(
                        f'removing {client_id} from active clients list due to exceeding of max local optimisation  '
                        f'steps ({len(active_clients_list)} clients left).')

        # Use selected clients for update.
        state, metrics = learning_process.next(state, sampled_train_data, sampled_client_states)
        metrics_logger.log({f'train/{metric}': value for metric, value in metrics['train'].items()}, step=round_num)

        # EARLY STOPPING
        # The early stopping strategy: stop the training if `val_loss` does not
        # decrease over a certain number of epochs.
        val_metrics = perform_evaluation(state.model, val_data_for_tff)

        # Logging
        metrics_logger.log({f'val/{metric}': value for metric, value in val_metrics.items()}, step=round_num)
        if accountant is not None:
            if cfg.dp.type == 'CDP':
                metrics_logger.log({'RDP/epsilon': accountant.get_privacy_spending_for_n_steps(round_num)[0]},
                                   step=round_num)
            else:
                average_privacy_spending = np.mean(
                    [single_accountant.get_privacy_spending_for_n_steps(
                        max_local_train_steps[client_id] - local_train_step_budget[client_id])[0]
                     for client_id, single_accountant in accountant.items()])
                metrics_logger.log({'RDP/epsilon': average_privacy_spending}, step=round_num)

        if cfg.early_stopping.enabled:
            if cfg.early_stopping.metric_name in val_metrics.keys():
                early_stopping_metric_value = val_metrics[cfg.early_stopping.metric_name]
            else:
                early_stopping_metric_value = tf.reduce_sum(
                    val_metrics['baseline_metrics'][cfg.early_stopping.metric_name] *
                    val_metrics['baseline_metrics']['num_test_examples']) / tf.cast(
                    tf.reduce_sum(val_metrics['baseline_metrics']['num_test_examples']), tf.float64)
                print(early_stopping_metric_value)

            if early_stopper.should_stop(early_stopping_metric_value, state):
                early_stopping_round = round_num - early_stopper.patience
                logging.info(f'Round {round_num} early stopping and resetting model weights to round{early_stopping_round}')
                metrics_logger.log({'early_stopping_round': early_stopping_round}, step=round_num)
                state = early_stopper.get_best_model_params()
                break
        if len(active_clients_list) == 0:
            logging.info(f'Round {round_num} early stopping due to empty client list (all local privacy budgets spent)')
            metrics_logger.log({'early_stopping_round': round_num}, step=round_num)
            break

    model = state.model
    metrics = perform_evaluation(model, test_data_for_tff)
    per_client_metrics = perform_per_client_evaluation(model)
    client_id_order = [cid.decode() for cid in per_client_metrics['context']]
    metrics_logger.log({f'test/{metric}': value for metric, value in metrics.items()}, step=(round_num + 1))
    metrics_logger.log({f'test_single_clients/{metric}': wandb.Histogram(value)
                        for metric, value in per_client_metrics['baseline_metrics'].items()}, step=(round_num + 1))
    metrics_logger.log({f'test_single_clients/{metric}_mean': np.mean(value)
                        for metric, value in per_client_metrics['baseline_metrics'].items()}, step=(round_num + 1))
    metrics_logger.log({f'test_single_clients/{metric}_std': np.std(value)
                        for metric, value in per_client_metrics['baseline_metrics'].items()}, step=(round_num + 1))
    pretty_print_results_dict(metrics, round_num + 1, '')
    logging.info(f'Per Client Metrics (order: {client_id_order}):')
    pretty_print_results_dict(per_client_metrics['baseline_metrics'], round_num + 1, '')

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
                metrics_logger.log(
                    {f'test_personalisation/{metric}_{client_id_order[client_index]}': metric_values[client_index]
                     for metric, metric_values in current_epoch_metrics.items()},
                    step=round_num + 1 + personalisation_epoch)
            early_stopping_client_indices = np.argwhere(early_stopping_rounds == personalisation_epoch)
            for client_index in early_stopping_client_indices:
                for metric in personalised_metrics:
                    personalised_metrics[metric][client_index] = current_epoch_metrics[metric][client_index]
        pretty_print_results_dict(personalised_metrics, round_num + 1, '')
        metrics_logger.log({f'test_single_clients/{metric}': wandb.Histogram(value)
                            for metric, value in personalised_metrics.items()}, step=(round_num + 2))
        metrics_logger.log({f'test_single_clients/{metric}_mean': np.mean(value)
                            for metric, value in personalised_metrics.items()}, step=(round_num + 2))
        metrics_logger.log({f'test_single_clients/{metric}_std': np.std(value)
                            for metric, value in personalised_metrics.items()}, step=(round_num + 2))

        for metric, value in personalised_metrics.items():
            per_client_df = pd.concat(
                [per_client_df, pd.Series(value, index=client_id_order, name=f'{metric}_personalised').to_frame().T])

    per_client_df = per_client_df.T
    logging.info(per_client_df)
    per_client_df.to_csv(f'{cfg.output_folder}/per_client_metrics.csv')
    metrics_logger.log({'test_single_clients/all_metrics': wandb.Table(dataframe=per_client_df)},
                       step=cfg.training.total_rounds)

    eval_model = model_fn()
    model.assign_weights_to(eval_model)

    # Manually predict test samples to compute curves
    test_data = test_data.create_tf_dataset_from_all_clients()
    labels = []
    pred_labels = []
    pred_probs = []
    for batch in test_data:
        labels.extend(batch['y'].numpy())
        predictions = eval_model(batch['x'], training=False)
        if 'calibrated_threshold' in metrics.keys():
            pred_labels.extend(tf.cast(predictions[:, 1] > metrics['calibrated_threshold'], tf.int32).numpy())
        else:
            pred_labels.extend(tf.argmax(predictions, axis=1).numpy())
        pred_probs.extend(predictions.numpy())
    metrics_logger.log({
        'test/conf_mat': wandb.plot.confusion_matrix(preds=pred_labels, y_true=labels),
        'test/pr_curve': wandb.plot.pr_curve(labels, pred_probs, interp_size=200, classes_to_plot=[1]),
        'test/roc_curve': wandb.plot.roc_curve(labels, pred_probs, classes_to_plot=[1])
    }, step=cfg.training.total_rounds)

    metrics_logger.write_metrics_to_file()

    return eval_model
