from sklearn.model_selection import train_test_split


def split_data(features, target, test_size, random_state=None):
    """
    Split the dataset into a training set and a test set.

    Parameters:
    - features: DataFrame containing the feature columns.
    - target: Series containing the target column.
    - test_size: float, the proportion of the dataset to include in the test split.
    - random_state: int, RandomState instance or None, optional (default=None)

    Returns:
    - X_train, X_test, y_train, y_test: The split datasets.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
