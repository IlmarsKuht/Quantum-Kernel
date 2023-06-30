from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import os
import numpy as np
import sys
import pandas

import time


#directory of the dataset is supposed to be passed as an argument
directory = sys.argv[1]
name = os.path.basename(directory)
dataset = np.array(pandas.read_csv(directory))

X = dataset[:, :-1]
Y = dataset[:, -1]

#fit the features between 0 and 1
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def classicalSVM(x_train, x_test, y_train, y_test):
    # List of all possible kernels
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']

    # Train and evaluate SVM models for each kernel
    for kernel in kernels:
        # Create an SVM classifier with the current kernel
        svm = SVC(kernel=kernel)

        # Train the SVM model
        start = time.time()
        svm.fit(x_train, y_train)
        end = time.time()
        # Evaluate the model on the test set
        accuracy = svm.score(x_test, y_test)

        # Print the accuracy for the current kernel
        print(f"Kernel: {kernel}, Accuracy: {accuracy:.3f}, Time: {end-start:.3f}")

classicalSVM(x_train, x_test, y_train, y_test)
