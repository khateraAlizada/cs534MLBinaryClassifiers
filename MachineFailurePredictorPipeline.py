import sklearn
import csv
from sklearn import svm
from sklearn.svm import SVC
from collections import Counter
from sklearn.metrics import f1_score
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from imblearn.under_sampling import RandomUnderSampler

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

    clf = svm.SVC(kernel='linear', C = 1).fit(X_train, y_train)
    print('Initial score')
    print(clf.score(X_train, y_train))

    #Tune hyperparameters

    param_grid = {'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                  'C' : [.0001, .001, .01, .1, 1, 10]}

    print('Tuning Hyperparameters using 5 fold cross validation')
    classifier = svm.SVC()
    clf = GridSearchCV(classifier, param_grid, cv=5, scoring= 'f1')

    print('fitting tuned hyperparameters')
    clf.fit(X_train, y_train)

    print('Best parameters', clf.best_params_)
    best_estimator = clf.best_estimator_
    y_pred = best_estimator.predict(X_test)

    print('f1 score test: ', f1_score(y_test, y_pred, average='weighted'))


if __name__ == "__main__":
    main()