from sklearn.tree import DecisionTreeClassifier


def train_decision_tree(X_train, y_train, random_state=42):
    """
    Train a Decision Tree Classifier.

    Parameters:
    - X_train: training features.
    - y_train: training target variable.
    - random_state: int, controls the randomness of the estimator.

    Returns:
    - decision_tree_clf: trained Decision Tree Classifier.
    """
    decision_tree_clf = DecisionTreeClassifier(random_state=random_state)
    decision_tree_clf.fit(X_train, y_train)
    return decision_tree_clf
