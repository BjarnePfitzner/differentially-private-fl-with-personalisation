import logging

from hydra.utils import instantiate

from src.data.support2.support2_dataset import Support2Dataset
from src.data.preprocess import get_preprocessed_data


def get_data_from_name(name):
    if name == 'support2':
        return Support2Dataset()
    else:
        raise NotImplementedError


def print_class_imbalance(Y, labels):
    logging.info(f'Class distributions for {len(Y)} data points:')
    for y_col in Y:
        if y_col not in labels:
            continue
        logging.info(f'Endpoint {y_col}:')
        abs_value_counts = Y[y_col].value_counts()
        rel_value_counts = Y[y_col].value_counts(normalize=True)
        for i in range(len(abs_value_counts.index)):
            logging.info(f'\tClass "{abs_value_counts.index[i]}":\t{abs_value_counts.iloc[i]} ({rel_value_counts.iloc[i]:.3f})')
        logging.info('\n')


def load_dataset(dataset_cfg, split_col=None):
    # Get DataInformation object for the specified task
    data = get_data_from_name(dataset_cfg.name)

    # Parse data
    data.parse(drop_columns=dataset_cfg.drop_features,
               feature_set=dataset_cfg.feature_set,
               drop_missing_value=dataset_cfg.drop_rows_missing_col_fraction,
               split_col=split_col)

    # Preprocess data
    X, Y = get_preprocessed_data(data,
                                 fs_operations=dataset_cfg.fs_operations,
                                 missing_threshold=dataset_cfg.drop_cols_missing_row_fraction,
                                 correlation_threshold=dataset_cfg.correlation_threshold,
                                 imputer=instantiate(dataset_cfg.imputer))

    return X, Y
