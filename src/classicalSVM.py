import os
import argparse
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from auxilary.data import load_dataset

# Parse the arguments
parser = argparse.ArgumentParser(description="classical SVM")
parser.add_argument('--dataset_dir', type=str, required=True, help='Directory of the dataset')
args = parser.parse_args()
dataset_dir = args.dataset_dir

name = os.path.splitext(os.path.basename(dataset_dir))[0]
print(f"Classical SVM on dataset {name}")

# Load and preprocess the dataset
x_train, x_test, y_train, y_test = load_dataset(dataset_dir)

# Standardize the features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Train and evaluate SVM models for each kernel and seed
seeds = [42, 1, 7]  # Different seeds for reproducibility
for seed in seeds:
    print(f"\nTraining with seed {seed}")
    for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
        print(f"\nKernel: {kernel}")
        for param_grid in tuned_parameters:
            # Only search for the current kernel
            if param_grid['kernel'][0] == kernel:
                svm = GridSearchCV(SVC(random_state=seed), param_grid, cv=5)
                svm.fit(x_train, y_train)
                print(f"Best parameters set found on development set: {svm.best_params_}")
                print(f"Grid scores on development set:")
                means = svm.cv_results_['mean_test_score']
                stds = svm.cv_results_['std_test_score']
                for mean, std, params in zip(means, stds, svm.cv_results_['params']):
                    print(f"{mean:.3f} (+/-{std * 2:.03f}) for {params}")
                
                accuracy = svm.score(x_test, y_test)
                print(f"Accuracy with best parameters: {accuracy:.3f}")
