from collections import Counter
from sklearn.datasets import make_classification
from imblearn.under_sampling import RandomUnderSampler
import csv


def main():
    X = []
    Y = []

    # Read in the CSV file
    with open("ai4i2020.csv", 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        for row in csvreader:
            X.append(row)
            Y.append(int(row[8]))
    print(header)
    print(Counter(Y))  # Before the filtering

    # Use RandomUnderSampler to get 339 datapoints with label 0 and 1
    sampler = RandomUnderSampler(sampling_strategy={0: 339, 1: 339})
    X_resampled, y_resampled = sampler.fit_resample(X, Y)

    print(Counter(y_resampled))  # After the filtering

    resampled_rows = [header] + [list(X_resampled[i]) + [y_resampled[i]] for i in range(len(X_resampled))]

    # Printing the resampled data
    for row in resampled_rows:
        print(row)


if __name__ == "__main__":
    main()
