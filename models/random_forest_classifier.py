from sklearn.ensemble import RandomForestClassifier  # Random Forest Classifier


def train_random_forest(X_train, y_train, random_state):
    """
    Train a Random Forest Classifier.

    Parameters:
    - X_train: training features.
    - y_train: training target variable.
    - random_state: int, controls the randomness of the estimator.

    Returns:
    - random_forest_clf: Trained Random Forest Classifier.
    """
    # Initialize the Random Forest Classifier
    random_forest_clf = RandomForestClassifier(random_state=random_state)

    # Train the classifier on the training data
    random_forest_clf.fit(X_train, y_train)
    return random_forest_clf
