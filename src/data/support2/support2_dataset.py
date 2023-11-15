import os
import logging

from ucimlrepo import fetch_ucirepo

from src.data.abstract_dataset import Dataset


#DATA_PATH = os.path.abspath("./src/data/support2/support2_data.csv")
DATA_INFOS_PATH = os.path.abspath("./src/data/support2/support2_dataset_infos.csv")


class Support2Dataset(Dataset):
    def __init__(self):
        super(Support2Dataset, self).__init__(None, DATA_INFOS_PATH)

    def read_csv(self):
        # fetch dataset
        support2 = fetch_ucirepo(id=880)

        # metadata
        logging.info(support2.metadata)

        # variable information
        logging.info(support2.variables)
        return support2.data.original
