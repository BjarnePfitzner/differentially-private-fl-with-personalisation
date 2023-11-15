import collections
from functools import partial
import logging

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
import tensorflow_federated as tff
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold


def _element_fn(X, y):
    return collections.OrderedDict(
        x=X, y=y)


def _get_class_weights(y: pd.Series, n_classes=2):
    return [len(y) / (n_classes * sum(y == cls)) for cls in range(n_classes)]


def _get_kfold_data_partition_from_seed_for_final_eval(X, y, train_fraction, seed):
    n_splits = int(1 / (1 - train_fraction))
    logging.debug(f'Splitting data into {n_splits} folds, choosing fold {(seed % n_splits) + 1}')
    logging.debug(f'Using {"first" if seed // n_splits == 0 else "second"} half of val_test for testing')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for i, (train_idx, val_test_idx) in enumerate(skf.split(X, y)):
        if i == seed % n_splits:
            train_X, val_test_X = X.iloc[train_idx], X.iloc[val_test_idx]
            train_y, val_test_y = y.iloc[train_idx], y.iloc[val_test_idx]
            break

    # catch error if number of positive class samples < n_splits * n_classes,
    # i.e., we don't have 2 positive class samples to split into val and test
    if len(val_test_y[val_test_y==1]) < 2:
        # randomly select positive sample from train data to add to val_test data
        pos_sample_ids = train_y[train_y==1].index
        random_sample_id = np.random.choice(pos_sample_ids)
        logging.debug(f'chose sample {random_sample_id} to add to test data')
        val_test_y = pd.concat([val_test_y, pd.Series(train_y.loc[random_sample_id], index=[random_sample_id])], axis=0)
        val_test_X = pd.concat([val_test_X, train_X.loc[random_sample_id].to_frame().T], axis=0)
        train_y.drop(index=random_sample_id, inplace=True)
        train_X.drop(index=random_sample_id, inplace=True)
    val_X, test_X, val_y, test_y = train_test_split(
        val_test_X, val_test_y,
        train_size=0.5,
        stratify=val_test_y,
        random_state=42
    )
    if seed // n_splits == 1:
        logging.debug('swapping val and test')
        val_X, val_y, test_X, test_y = test_X, test_y, val_X, val_y

    return train_X, train_y, val_X, val_y, test_X, test_y


def _get_kfold_data_partition_from_seed(X, y, train_fraction, seed):
    n_splits = int(1 / (1 - train_fraction))
    logging.debug(f'Splitting data into {n_splits} folds, choosing fold {(seed % n_splits) + 1}')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + (seed // n_splits))

    for i, (train_idx, val_test_idx) in enumerate(skf.split(X, y)):
        if i == seed % n_splits:
            train_X, val_test_X = X.iloc[train_idx], X.iloc[val_test_idx]
            train_y, val_test_y = y.iloc[train_idx], y.iloc[val_test_idx]
            break

    # catch error if number of positive class samples < n_splits * n_classes,
    # i.e., we don't have 2 positive class samples to split into val and test
    if len(val_test_y[val_test_y==1]) < 2:
        split1_X, split2_X, split1_y, split2_y = train_test_split(
            val_test_X, val_test_y,
            train_size=0.5,
            random_state=42
        )
        # Use the split with positive sample as test data
        if len(split1_y[split1_y==1]) > 0:
            test_X, test_y, val_X, val_y = split1_X, split1_y, split2_X, split2_y
        else:
            test_X, test_y, val_X, val_y = split2_X, split2_y, split1_X, split1_y
    else:
        val_X, test_X, val_y, test_y = train_test_split(
            val_test_X, val_test_y,
            train_size=0.5,
            stratify=val_test_y,
            random_state=42
        )

    return train_X, train_y, val_X, val_y, test_X, test_y


def prepare_kfold_centralised_dss(X, y, cfg):
    # unused
    all_datas = []
    class_weights = _get_class_weights(y)
    logging.info(f'Class weights: {class_weights}')

    train_val_X, test_X, train_val_y, test_y = train_test_split(
        X, y,
        train_size=(cfg.dataset.train_fraction + 1) / 2,
        stratify=y,
        random_state=cfg.zeed
    )
    skf = StratifiedKFold(n_splits=cfg.dataset.k_folds, shuffle=True, random_state=cfg.zeed)
    for train_idx, val_idx in skf.split(train_val_X, train_val_y):
        train_X, val_X = train_val_X.iloc[train_idx], train_val_X.iloc[val_idx]
        train_y, val_y = train_val_y.iloc[train_idx], train_val_y.iloc[val_idx]
        all_datas.append((train_X, train_y, val_X, val_y, test_X, test_y))

    train_ds = []
    val_ds = []
    test_ds = []
    for train_X, train_y, val_X, val_y, test_X, test_y in all_datas:
        train_ds.append(tf.data.Dataset.from_tensor_slices((train_X, train_y))
                        .map(_element_fn)
                        .shuffle(len(train_y), reshuffle_each_iteration=True, seed=cfg.zeed)
                        .batch(cfg.training.batch_size, drop_remainder=(cfg.dp.type == 'LDP'))
                        .prefetch(AUTOTUNE))
        val_ds.append(tf.data.Dataset.from_tensor_slices((val_X, val_y))
                      .map(_element_fn)
                      .shuffle(len(val_y), reshuffle_each_iteration=False, seed=cfg.zeed)
                      .batch(cfg.evaluation.batch_size)
                      .prefetch(AUTOTUNE))
        test_ds.append(tf.data.Dataset.from_tensor_slices((test_X, test_y))
                       .map(_element_fn)
                       .shuffle(len(test_y), reshuffle_each_iteration=False, seed=cfg.zeed)
                       .batch(cfg.evaluation.batch_size)
                       .prefetch(AUTOTUNE))

    if cfg.dp.type == 'LDP':
        train_size = int(len(train_y) / cfg.training.batch_size) * cfg.training.batch_size
    else:
        train_size = len(train_y)

    return train_ds, val_ds, test_ds, train_size, class_weights


def prepare_centralised_ds(X, y, cfg):
    class_weights = _get_class_weights(y)
    logging.info(f'Class weights: {class_weights}')

    if cfg.final_eval:
        train_X, train_y, val_X, val_y, test_X, test_y = _get_kfold_data_partition_from_seed_for_final_eval(
            X, y, cfg.dataset.train_fraction, cfg.zeed)
    else:
        train_X, train_y, val_X, val_y, test_X, test_y = _get_kfold_data_partition_from_seed(X, y,
                                                                                             cfg.dataset.train_fraction,
                                                                                             cfg.zeed)

    train_ds = (tf.data.Dataset.from_tensor_slices((train_X, train_y))
                .map(_element_fn)
                .shuffle(len(train_y), reshuffle_each_iteration=True, seed=cfg.zeed)
                .batch(cfg.training.batch_size, drop_remainder=(cfg.dp.type == 'LDP'))
                .prefetch(AUTOTUNE))
    val_ds = (tf.data.Dataset.from_tensor_slices((val_X, val_y))
              .map(_element_fn)
              .shuffle(len(val_y), reshuffle_each_iteration=False, seed=cfg.zeed)
              .batch(cfg.evaluation.batch_size)
              .prefetch(AUTOTUNE))
    test_ds = (tf.data.Dataset.from_tensor_slices((test_X, test_y))
               .map(_element_fn)
               .shuffle(len(test_y), reshuffle_each_iteration=False, seed=cfg.zeed)
               .batch(cfg.evaluation.batch_size)
               .prefetch(AUTOTUNE))

    if cfg.dp.type == 'LDP':
        train_size = int(len(train_y) / cfg.training.batch_size) * cfg.training.batch_size
    else:
        train_size = len(train_y)

    return train_ds, val_ds, test_ds, train_size, class_weights


def prepare_federated_ds(X, y, cfg):
    if cfg.training.split_clients_by == 'random':
        client_ids = [f'client_{id}' for id in range(cfg.training.n_total_clients)]
        perm = np.random.permutation(len(X))
        perm_splits = np.array_split(perm, cfg.training.n_total_clients)
        client_data = {cid: (X.iloc[perm_splits[i]],
                             y.iloc[perm_splits[i]])
                       for i, cid in enumerate(client_ids)}
    elif cfg.training.split_clients_by in X.columns:
        client_values = X[cfg.training.split_clients_by].unique()
        client_ids = [cfg.training.split_clients_by + '_' + str(client_value) for client_value in client_values]
        client_data = {cid: (X[X[cfg.training.split_clients_by] == client_value],
                             y[X[cfg.training.split_clients_by] == client_value])
                       for cid, client_value in zip(client_ids, client_values)}
    else:
        client_ids = list(X.columns[X.columns.str.startswith(cfg.training.split_clients_by)])
        client_data = {cid: (X[X[cid] == 1],
                             y[X[cid] == 1])
                       for cid in client_ids}
    # Potentially drop columns
    if cfg.training.drop_split_columns:
        client_data = {cid: (client_data[0].drop(columns=client_data[0].columns[client_data[0].columns.str.startswith(cfg.training.split_clients_by)]),
                             client_data[1])
                       for cid, client_data in client_data.items()}
    cfg.training.n_total_clients = len(client_ids)

    client_dataset_sizes = [len(y) for _, y in client_data.values()]
    logging.info(f'Found {len(client_ids)} clients with {client_dataset_sizes} samples')
    logging.info(f'Per-client positive sample fraction:')
    for cid, single_client_data in client_data.items():
        the_y = single_client_data[1]
        logging.info(f'\t{cid}: {len(the_y[the_y==1]) / len(the_y):.3f} ({len(the_y[the_y==1])})')

    n_input_features = len(client_data[list(client_data.keys())[0]][0].columns)

    client_train_data, client_val_data, client_test_data = {}, {}, {}
    client_train_sizes = {}
    client_class_weights = {}
    logging.info('Client class weights:')
    for cid, single_client_data in client_data.items():
        if cfg.final_eval:
            client_train_X, client_train_y, client_val_X, client_val_y, client_test_X, client_test_y = (
                _get_kfold_data_partition_from_seed_for_final_eval(single_client_data[0], single_client_data[1],
                                                                   cfg.dataset.train_fraction, cfg.zeed))
        else:
            client_train_X, client_train_y, client_val_X, client_val_y, client_test_X, client_test_y = (
                _get_kfold_data_partition_from_seed(single_client_data[0], single_client_data[1],
                                                    cfg.dataset.train_fraction, cfg.zeed))
        client_train_data[cid] = client_train_X, client_train_y
        client_val_data[cid] = client_val_X, client_val_y
        client_test_data[cid] = client_test_X, client_test_y
        client_train_sizes[cid] = len(client_train_y)
        client_class_weights[cid] = _get_class_weights(single_client_data[1])
        logging.info(f'\t{cid}: {[round(weight, 3) for weight in client_class_weights[cid]]}')

    def get_client_ds_from_dict_with_id(data_dict, client_id):
        return tf.data.Dataset.from_tensor_slices(data_dict[client_id])

    # Preprocess dataset
    def preprocess_train_dataset(dataset):
        return (dataset
                .map(_element_fn)
                .shuffle(buffer_size=max(client_dataset_sizes))
                .repeat(cfg.training.local_epochs)
                .batch(cfg.training.batch_size, drop_remainder=(cfg.dp.type == 'LDP'))
                .prefetch(AUTOTUNE)
                )

    def preprocess_test_dataset(dataset):
        return (dataset
                .map(_element_fn)
                .batch(cfg.evaluation.batch_size)
                .prefetch(AUTOTUNE)
                )

    # prepare val dataset
    if cfg.training.single_client_validation:
        largest_val_set_client_id = max(client_val_data, key=lambda x: len(client_val_data[x][1]))
        logging.info(f'Using client {largest_val_set_client_id} for val set')
        for cid in client_data.keys():
            if cid == largest_val_set_client_id:
                tff_val_ds = tff.simulation.datasets.ClientData.from_clients_and_fn(
                                [cid], partial(get_client_ds_from_dict_with_id, client_val_data))
            else:
                #client_test_data[cid] = (client_test_data[cid][0].append(client_val_data[cid][0]),
                #                         client_test_data[cid][1].append(client_val_data[cid][1]))
                client_train_data[cid] = (client_train_data[cid][0].append(client_val_data[cid][0]),
                                          client_train_data[cid][1].append(client_val_data[cid][1]))
                client_train_sizes[cid] += len(client_val_data[cid][1])
    else:
        tff_val_ds = tff.simulation.datasets.ClientData.from_clients_and_fn(
            client_ids, partial(get_client_ds_from_dict_with_id, client_val_data))
    val_ds = tff_val_ds.preprocess(preprocess_test_dataset)

    # prepare federated train dataset
    tff_train_ds = tff.simulation.datasets.ClientData.from_clients_and_fn(
        client_ids, partial(get_client_ds_from_dict_with_id, client_train_data))
    train_ds = tff_train_ds.preprocess(preprocess_train_dataset)

    # Prepare test set
    tff_test_ds = tff.simulation.datasets.ClientData.from_clients_and_fn(
        client_ids, partial(get_client_ds_from_dict_with_id, client_test_data))
    test_ds = tff_test_ds.preprocess(preprocess_test_dataset)

    return train_ds, val_ds, test_ds, client_train_sizes, client_class_weights, n_input_features
