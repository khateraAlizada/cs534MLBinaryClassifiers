from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import RandomUnderSampler
import csv


def main():
    X = []  # UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
    Y = []  # Machine Failure

    # Read in the CSV file
    with open("ai4i2020.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        # Read each row and add the values to the arrays
        for row in csvreader:
            # Convert the string values to appropriate data types
            udi = int(row[0])
            # product_id = row[1]
            # Convert the 'Type' column to a numerical value (e.g. 'L' -> 1, 'M' -> 2, 'H' -> 3)
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

    # Printing split results - TEMPORARY PRINTS
    print("X_train:", X_train)
    print("X_test:", X_test)
    print("Y_train:", Y_train)
    print("Y_test:", Y_test)

    random_forest_results = RandomForest(X_train, X_test, Y_train, Y_test)

    print(random_forest_results[0])
    print(random_forest_results[1])
    print(random_forest_results[2])


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
                     + "\n    F1 Score on test Data: " + str(f1_score(Y_test, Y_pred))

    final_f1_score = str(f1_score(Y_test, Y_pred))

    return [training_result, testing_result, final_f1_score]


if __name__ == "__main__":
    main()
