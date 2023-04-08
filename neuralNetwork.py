from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import csv

def main():
   # X = []  # UDI,Product ID,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
    X = []  # UDI,Type,Air temperature [K],Process temperature [K],Rotational speed [rpm],Torque [Nm],Tool wear [min]
    Y = []  # Machine Failure

    # Read in the CSV file
    with open("ai4i2020.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
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
            X.append(
                [udi, typ, air_temp, process_temp, rotational_speed, torque, tool_wear, twf, hdf, pwf, osf,
                 rnf])
            Y.append(machine_failure)
        #for row in csvreader:
            # Convert the values to float and append to the X list
            #X.append([float(i) for i in row[3:8]])
            #Y.append(int(row[8]))

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
        'hidden_layer_sizes': [(10,), (20,), (30,)],
        'alpha': [0.0001, 0.001, 0.01],
        'learning_rate_init': [0.001, 0.01, 0.1]
    }

    # Initialize the MLPClassifier
    mlp = MLPClassifier(max_iter=1000)

    # Use GridSearchCV to perform 5-fold cross-validation and fine-tune hyperparameters
    cv_results = GridSearchCV(mlp, param_grid, cv=5, scoring='f1')
    cv_results.fit(X_train, Y_train)

    # Print the results of the grid search
    print("Best parameters:", cv_results.best_params_)
    print("Best F1-score:", cv_results.best_score_)

    # Use the best estimator to make predictions on the test set
    best_estimator = cv_results.best_estimator_
    Y_pred = best_estimator.predict(X_test)

    # Calculate the F1-score of the best estimator on the test set
    f1 = f1_score(Y_test, Y_pred)
    print("F1-score on test data:", f1)

    # Calculate the F1-score on the 5-fold Cross Validation on Training Data (70%)
    f1_cv = cv_results.best_score_
    print("F1-score on 5-fold CV on Training Data (70%):", f1_cv)
if __name__ == "__main__":
    main()