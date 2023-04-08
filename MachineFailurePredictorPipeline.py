from collections import Counter

import sklearn
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier
import csv
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

def main():
    X = []
    Y = []


    #df = pd.read_csv("ai4i2020.csv")
    #print(df)

    # Read in the CSV file
    with open("ai4i2020.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            for character in 'LMH':
                row[1] = row[1].replace(character, '')
            del row[2]
            X.append(row)
            Y.append(int(row[7]))
    print(header)
    print(Counter(Y))  # Before the filtering

    # Use RandomUnderSampler to get 339 datapoints with label 0 and 1
    sampler = RandomUnderSampler(sampling_strategy={0: 339, 1: 339})
    X_resampled, Y_resampled = sampler.fit_resample(X, Y)

    print(Counter(Y_resampled))  # After the filtering

    resampled_rows = [header] + [list(X_resampled[i]) + [Y_resampled[i]] for i in range(len(X_resampled))]

    # Printing the resampled data
    #for row in resampled_rows:
     #   print(row)

    #train-test split
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X_resampled,Y_resampled, test_size=.3, train_size=.7, random_state=None, shuffle=True,
                                             stratify=None)

    #print(type(X_train))
    #print(X_train[1])
    #print(X_train[2])
    clf = BaggingClassifier(estimator=SVC(),
                            n_estimators=10, max_samples= 10, max_features=10, random_state=0).fit(X_train, y_train)
    print('Initial score')
    print(clf.score(X_train, y_train))

    #Tune hyperparameters
    print('Tuning Hyperparameters')
    param_grid = {'n_estimators': [1, 10, 25, 50],
                  'max_samples' : [.01, .1, 1, 10, 50, 100],
                  'max_features': [.02, .04, .06, .08, .1, .12]}

    bag = BaggingClassifier(estimator=SVC())
    clf = GridSearchCV(bag, param_grid)


    print('fitting tuned hyperparameters')
    clf.fit(X_train, y_train)

    print('Best parameters', clf.best_params_)
    y_pred = clf.predict(X_train)
    print('f1 score train: ', f1_score(y_train, y_pred, average='weighted'))

    #5 fold cross validation score on training data
    scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='f1_weighted')
    print("Cross validation score", scores)
    print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

    y_pred = clf.predict(X_test)

    print('f1 score test: ', f1_score(y_test, y_pred, average='weighted'))


if __name__ == "__main__":
    main()