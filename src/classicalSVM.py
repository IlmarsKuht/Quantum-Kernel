from sklearn.svm import SVC

from aux.data import load_dataset

import os
import sys

import time


#directory of the dataset is supposed to be passed as an argument
dataset_dir = sys.argv[1]
name = os.path.splitext(os.path.basename(dataset_dir))[0]
print(f"Classical SVM on dataset {name}")

x_train, x_test, y_train, y_test = load_dataset(dataset_dir)


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