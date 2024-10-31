from sklearn.svm import SVC  # Support Vector Classifier


def train_svm(X_train, y_train, random_state):
    """
    Train a Support Vector Classifier.

    Parameters:
    - X_train: training features.
    - y_train: training target variable.
    - random_state: int, controls the randomness of the estimator.

    Returns:
    - svm_clf: Trained Support Vector Classifier.
    """
    # Initialize the SVM classifier
    # The default kernel is 'rbf' which is the one we used, other kernal we tried are 'linear', 'poly', etc.
    svm_clf = SVC(random_state=random_state)

    # Train the classifier on the training data
    svm_clf.fit(X_train, y_train)
    return svm_clf
