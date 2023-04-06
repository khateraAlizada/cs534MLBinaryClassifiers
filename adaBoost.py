from collections import Counter
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
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

    # Initialize the AdaBoostClassifier with a DecisionTreeClassifier as the base estimator and set the number of estimators
    clf = AdaBoostClassifier(base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50)

    # Fit the classifier on the training data
    clf.fit(X_train, Y_train)

    # Make predictions on the testing data
    Y_pred = clf.predict(X_test)

    # Calculate the accuracy of the model
    accuracy = accuracy_score(Y_test, Y_pred)
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()