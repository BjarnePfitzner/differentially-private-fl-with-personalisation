import logging


str_replacement_dict = {
    'loss': 'Loss',
    'accuracy': 'Acc',
    'auprc': 'AUPRC',
    'auroc': 'AUROC',
    'f1': 'F1',
    'calibrated_f1': 'F1 (Cal)',
    'calibrated_recall': 'Rec (Cal)',
    'calibrated_precision': 'Prc (Cal)',
    'calibrated_threshold': 'Thresh',
    'f2': 'F2',
    'calibrated_f2': 'F2 (Cal)',
    'calibrated_f2_recall': 'F2 Rec (Cal)',
    'calibrated_f2_precision': 'F2 Prc (Cal)',
    'calibrated_f2_threshold': 'F2 Thresh'
}


def _get_formatted_string_from_entry(entry):
    if isinstance(entry, list):
        if isinstance(entry[0], float):
            return [f'{list_entry:.3f}' for list_entry in entry]
    elif isinstance(entry, float):
        return f'{entry:.3f}'
    else:
        return str(entry)


def pretty_print_results_dict(results_dict, round, prefix='test/'):
    pretty_dict = {str_replacement_dict.get(key.replace(prefix, ''), key.replace(prefix, '')): value
                   for key, value in results_dict.items()}
    logging.info(f'Round {round:3d} - Loss:       {_get_formatted_string_from_entry(pretty_dict["Loss"])}')
    for metric_name, metric_value in sorted(pretty_dict.items()):
        if metric_name == 'Loss' or metric_name.startswith('_'):
            continue
        logging.info(f'            {metric_name + ":":11} {_get_formatted_string_from_entry(metric_value)}')
