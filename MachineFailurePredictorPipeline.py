from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from imblearn.under_sampling import RandomUnderSampler
import csv

from sklearn.svm import SVC
from collections import Counter
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import cross_val_score


def main():
    X = []  # UDI,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
    Y = []  # Machine Failure

    # Read in the CSV file
    with open("ai4i2020.csv", 'r') as file:
        csvreader = csv.reader(file)
        next(csvreader)

        # Read each row and add the values to the arrays
        for row in csvreader:
            # Convert the string values to appropriate data types
            udi = int(row[0])
            # product_id = row[1] # Omitted product ID
            # Converting Type to integer equivalent: 'L' -> 1, 'M' -> 2, 'H' -> 3)
            if row[2] == 'L':
                typ = 1
            elif row[2] == 'M':
                typ = 2
            else:
                typ = 3
            air_temp = float(row[3])
            process_temp = float(row[4])
            rotational_speed = int(row[5])
            torque = float(row[6])
            tool_wear = int(row[7])
            machine_failure = int(row[8])
            twf = int(row[9])
            hdf = int(row[10])
            pwf = int(row[11])
            osf = int(row[12])
            rnf = int(row[13])

            # Add the values to the arrays
            X.append([udi, typ, air_temp, process_temp, rotational_speed, torque, tool_wear, twf, hdf, pwf, osf, rnf])
            Y.append(machine_failure)

    # Use RandomUnderSampler to get 339 datapoints with label 0 and 1
    sampler = RandomUnderSampler(sampling_strategy={0: 339, 1: 339})
    X_resampled, Y_resampled = sampler.fit_resample(X, Y)

    # Split the data into 70% training and 30% testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, Y_resampled, test_size=0.3, random_state=42)

    # Printing split results
    # print("X_train:", X_train)
    # print("X_test:", X_test)
    # print("Y_train:", Y_train)
    # print("Y_test:", Y_test)

    artificial_neural_networks_results = ArtificialNeuralNetworks(X_train, X_test, Y_train, Y_test)
    support_vector_machine_results = SupportVectorMachine(X_train, X_test, Y_train, Y_test)
    bagging_classifier_results = BaggingClassifierModel(X_train, X_test, Y_train, Y_test)
    ada_boost_results = AdaBoost(X_train, X_test, Y_train, Y_test)
    random_forest_results = RandomForest(X_train, X_test, Y_train, Y_test)

    print("\nBest parameters for each ML model:\n")
    print(artificial_neural_networks_results[0])
    print(support_vector_machine_results[0])
    print(bagging_classifier_results[0])
    print(ada_boost_results[0])
    print(random_forest_results[0])

    print("\nPerformance on testing data:\n")
    print(artificial_neural_networks_results[1])
    print(support_vector_machine_results[1])
    print(bagging_classifier_results[1])
    print(ada_boost_results[1])
    print(random_forest_results[1])

    ann_f1 = float(artificial_neural_networks_results[2])
    svm_f1 = float(support_vector_machine_results[2])
    bc_f1 = float(bagging_classifier_results[2])
    ab_f1 = float(ada_boost_results[2])
    rf_f1 = float(random_forest_results[2])

    max_f1_score = max(ann_f1, svm_f1, bc_f1, ab_f1, rf_f1)

    print("\nBest ML Model based on F1 scores: ")

    if max_f1_score == ann_f1:
        print("Artificial Neural Networks with an F1 score of " + artificial_neural_networks_results[2])
    elif max_f1_score == svm_f1:
        print("Support Vector Machine with an F1 score of " + support_vector_machine_results[2])
    elif max_f1_score == bc_f1:
        print("Bagging Classifier with an F1 score of " + bagging_classifier_results[2])
    elif max_f1_score == ab_f1:
        print("Ada Boost with an F1 score of " + ada_boost_results[2])
    else:
        print("Random Forest with an F1 score of " + random_forest_results[2])


def ArtificialNeuralNetworks(X_train, X_test, Y_train, Y_test):
    param_grid = {
        'hidden_layer_sizes': [(10,), (50,), (100,), (10, 10), (50, 50), (100, 100)],
        'activation': ['relu', 'logistic', 'tanh']
    }

    # Initialize the MLPClassifier
    mlp = MLPClassifier(max_iter=1000)

    # Use GridSearchCV to perform 5-fold cross-validation and fine-tune hyperparameters
    cv_results = GridSearchCV(mlp, param_grid, cv=5, scoring='f1')
    cv_results.fit(X_train, Y_train)

    # Get the best parameters and f1 score on training data
    training_result = "Artificial Neural Networks: " \
                      + "\n    Best Parameter Set: " + str(cv_results.best_params_) \
                      + "\n    F1 score on training data: " + str(cv_results.best_score_)

    # Use the best estimator to make predictions on the test set
    best_estimator = cv_results.best_estimator_
    Y_pred = best_estimator.predict(X_test)

    testing_result = "Artificial Neural Networks: " \
                     + "\n    Best Parameter Set: " + str(cv_results.best_params_) \
                     + "\n    F1 Score on test data: " + str(f1_score(Y_test, Y_pred))

    # Get the F1 score on the testing data for later comparison
    final_f1_score = str(f1_score(Y_test, Y_pred))

    return [training_result, testing_result, final_f1_score]


def SupportVectorMachine(X_train, X_test, Y_train, Y_test):
    clf = svm.SVC(kernel='linear', C=1).fit(X_train, Y_train)
    # print('Initial score')
    # print(clf.score(X_train, Y_train))

    # Tune hyperparameters
    param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': [.0001, .001, .01, .1, 1, 10]}

    # print('Tuning Hyperparameters using 5-fold cross validation')
    classifier = svm.SVC()
    clf = GridSearchCV(classifier, param_grid, cv=5, scoring='f1')

    # print('fitting tuned hyperparameters')
    clf.fit(X_train, Y_train)

    # print('Best parameters', clf.best_params_)
    best_estimator = clf.best_estimator_
    y_pred = best_estimator.predict(X_test)

    # Get the best parameters and f1 score on training data
    training_result = "Support Vector Machine: " \
                      + "\n    Best Parameter Set: " + str(clf.best_params_) \
                      + "\n    F1 score on training data: " + str(clf.best_score_)

    testing_result = "Support Vector Machine: " \
                     + "\n    Best Parameter Set: " + str(clf.best_params_) \
                     + "\n    F1 Score on test data: " + str(f1_score(Y_test, y_pred, average='weighted'))

    # Get the F1 score on the testing data for later comparison
    final_f1_score = str(f1_score(Y_test, y_pred, average='weighted'))

    return [training_result, testing_result, final_f1_score]


def BaggingClassifierModel(X_train, X_test, Y_train, Y_test):
    # clf = BaggingClassifier(estimator=SVC(), n_estimators=10, max_samples=10, max_features=10, random_state=0).fit(X_train, Y_train)
    # print('Initial score')
    # print(clf.score(X_train, y_train))

    # Tune hyperparameters
    # print('Tuning Hyperparameters with 5-fold cross validation')
    param_grid = {'n_estimators': [1, 10, 25, 50],
                  'max_samples': [.01, .1, 1, 10, 50, 100],
                  'max_features': [.02, .04, .06, .08, .1, .12]}

    bag = BaggingClassifier()
    clf = GridSearchCV(bag, param_grid, cv=5, scoring='f1')

    # print('fitting tuned hyperparameters')
    clf.fit(X_train, Y_train)
    best_estimator = clf.best_estimator_
    best_estimator.fit(X_train, Y_train)

    # print('Best parameters', clf.best_params_)
    y_pred_train = best_estimator.predict(X_train)
    # print('f1 score train: ', f1_score(Y_train, y_pred, average='weighted'))

    y_pred_test = best_estimator.predict(X_test)

    # Get the best parameters and f1 score on training data
    training_result = "Bagging Classifier: " \
                      + "\n    Best Parameter Set: " + str(clf.best_params_) \
                      + "\n    F1 score on training data: " + str(f1_score(Y_train, y_pred_train, average='weighted'))

    testing_result = "Bagging Classifier: " \
                     + "\n    Best Parameter Set: " + str(clf.best_params_) \
                     + "\n    F1 Score on test data: " + str(f1_score(Y_test, y_pred_test, average='weighted'))

    # Get the F1 score on the testing data for later comparison
    final_f1_score = str(f1_score(Y_test, y_pred_test, average='weighted'))

    return [training_result, testing_result, final_f1_score]


def AdaBoost(X_train, X_test, Y_train, Y_test):
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

    # Get the best parameters and f1 score on training data
    training_result = "Ada Boost: " \
                      + "\n    Best Parameter Set: " + str(cv_results.best_params_) \
                      + "\n    F1 score on training data: " + str(cv_results.best_score_)

    # Use the best estimator to make predictions on the test set
    best_estimator = cv_results.best_estimator_
    Y_pred = best_estimator.predict(X_test)

    testing_result = "Ada Boost: " \
                     + "\n    Best Parameter Set: " + str(cv_results.best_params_) \
                     + "\n    F1 Score on test data: " + str(f1_score(Y_test, Y_pred))

    # Get the F1 score on the testing data for later comparison
    final_f1_score = str(f1_score(Y_test, Y_pred))

    return [training_result, testing_result, final_f1_score]


def RandomForest(X_train, X_test, Y_train, Y_test):
    # Values to be explored
    param_grid = {
        'n_estimators': [300, 500],
        'criterion': ['entropy'],
        'max_features': ['log2'],
        'max_depth': [10, 20, 30],
        'max_samples': [0.5, 0.7, None]
    }

    # Default DecisionTreeClassifier values
    rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_features='sqrt', max_depth=None, max_samples=1.0, random_state=42)

    # Grid Search w/ 5-fold cross validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train, Y_train)

    # Get the best parameters and f1 score on training data
    training_result = "Random Forest: " \
                      + "\n    Best Parameter Set: " + str(grid_search.best_params_) \
                      + "\n    F1 score on training data: " + str(grid_search.best_score_)

    # Get the performance of the best parameter on test data
    best_rf = grid_search.best_estimator_
    Y_pred = best_rf.predict(X_test)

    testing_result = "Random Forest: " \
                     + "\n    Best Parameter Set: " + str(grid_search.best_params_) \
                     + "\n    F1 Score on test data: " + str(f1_score(Y_test, Y_pred))

    # Get the F1 score on the testing data for later comparison
    final_f1_score = str(f1_score(Y_test, Y_pred))

    return [training_result, testing_result, final_f1_score]


if __name__ == "__main__":
    main()

