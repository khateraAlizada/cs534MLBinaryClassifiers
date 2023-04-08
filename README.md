# CS 534 Group Assignment #3: ML Binary Classifiers

Code roughly takes about 5 minutes to execute. Below is a sample execution:

```
Best parameters for each ML model:

Artificial Neural Networks: 
    Best Parameter Set: {'activation': 'logistic', 'hidden_layer_sizes': (50, 50)}
    F1 score on training data: 0.8171723520326004
Support Vector Machine: 
    Best Parameter Set: {'C': 0.1, 'kernel': 'linear'}
    F1 score on training data: 0.926093750856366
Bagging Classifier: 
    Best Parameter Set: {'max_features': 0.02, 'max_samples': 0.1, 'n_estimators': 50}
    F1 score on training data: 0.8985822052087953
Ada Boost: 
    Best Parameter Set: {'learning_rate': 0.1, 'n_estimators': 50}
    F1 score on training data: 0.9833762886597939
Random Forest: 
    Best Parameter Set: {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'max_samples': None, 'n_estimators': 300}
    F1 score on training data: 0.9813565116768357

Performance on testing data:

Artificial Neural Networks: 
    Best Parameter Set: {'activation': 'logistic', 'hidden_layer_sizes': (50, 50)}
    F1 Score on test data: 0.8453608247422681
Support Vector Machine: 
    Best Parameter Set: {'C': 0.1, 'kernel': 'linear'}
    F1 Score on test data: 0.9509376549224745
Bagging Classifier: 
    Best Parameter Set: {'max_features': 0.02, 'max_samples': 0.1, 'n_estimators': 50}
    F1 Score on test data: 0.8432428051485903
Ada Boost: 
    Best Parameter Set: {'learning_rate': 0.1, 'n_estimators': 50}
    F1 Score on test data: 0.9946524064171123
Random Forest: 
    Best Parameter Set: {'criterion': 'entropy', 'max_depth': 10, 'max_features': 'log2', 'max_samples': None, 'n_estimators': 300}
    F1 Score on test data: 0.9946524064171123

Best ML Model based on F1 scores: 
Ada Boost with an F1 score of 0.9946524064171123

Process finished with exit code 0
```