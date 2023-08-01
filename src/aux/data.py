from typing import Tuple

import pandas
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_dataset(dataset_dir: str, test_size: float=0.2, normalize: bool=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """loads the file and splits into train and test data

    Args:
        dataset_dir (str): path to file
        test_size (float, optional): Defaults to 0.2.
        normalize (bool, optional): Normalize features between 0 and 1. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: train features, test features, train labels, test labels
    """
    dataset = np.array(pandas.read_csv(dataset_dir))

    X = dataset[:, :-1]
    Y = dataset[:, -1]

    if normalize:
        #fit the features between 0 and 1
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
    return x_train, x_test, y_train, y_test