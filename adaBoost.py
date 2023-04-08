from collections import Counter
from sklearn.model_selection import train_test_split, GridSearchCV

from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
import csv

def main():
    X = []  # UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
    Y = []  # Machine Failure

    # Read in the CSV file
    with open("ai4i2020.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            # Convert the values to float and append to the X list
            X.append([float(i) for i in row[3:8]])
            Y.append(int(row[8]))
    print(header)  # CSV Header - TEMPORARY PRINT
    print(Counter(Y))  # Before the filtering - TEMPORARY PRINT

    # Use RandomUnderSampler to get 339 datapoints with label 0 and 1
    sampler = RandomUnderSampler(sampling_strategy={0: 339, 1: 339})
    X_resampled, Y_resampled = sampler.fit_resample(X, Y)

    print(Counter(Y_resampled))  # After the filtering - TEMPORARY PRINT

    resampled_rows = [header[3:8] + ['Machine Failure']] + [[str(val) for val in X_resampled[i]] + [Y_resampled[i]] for i in range(len(X_resampled))]

    # Printing the resampled data - TEMPORARY PRINT
    for row in resampled_rows:
        print(row)

    # Split the data into 70% training and 30% testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)
    # Define the hyperparameter grid to search over
    param_grid = {
        'n_estimators': [10, 50, 100],
        'learning_rate': [0.1, 0.5, 1.0]
    }

    # Initialize the AdaBoostClassifier with a DecisionTreeClassifier as the base estimator
    base_estimator = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(base_estimator=base_estimator)

    # Use GridSearchCV to perform 5-fold cross-validation and fine-tune hyperparameters
    cv_results = GridSearchCV(clf, param_grid, cv=5, scoring='f1')
    cv_results.fit(X_train, Y_train)

    # Print the results of the grid search
    print("Best parameters:", cv_results.best_params_)
    print("Best F1-score on Training data:", cv_results.best_score_)

    # Use the best estimator to make predictions on the test set
    best_estimator = cv_results.best_estimator_
    Y_pred = best_estimator.predict(X_test)

    # Calculate the F1-score of the best estimator on the test set
    f1 = f1_score(Y_test, Y_pred)
    print("F1-score on test set:", f1)

    # Initialize the AdaBoostClassifier with a DecisionTreeClassifier as the base estimator and set the number of estimators
    ##clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)

    # Fit the classifier on the training data
    ##clf.fit(X_train, Y_train)

    # Make predictions on the testing data
    ##Y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    ##accuracy = accuracy_score(Y_test, Y_pred)
    ##print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()