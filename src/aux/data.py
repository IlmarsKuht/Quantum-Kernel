import pandas
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_dataset(dataset_dir, test_size=0.2):
        dataset = np.array(pandas.read_csv(dataset_dir))

        X = dataset[:, :-1]
        Y = dataset[:, -1]

        #fit the features between 0 and 1
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size)
        return x_train, x_test, y_train, y_test