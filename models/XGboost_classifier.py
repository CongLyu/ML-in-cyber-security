import xgboost as xgb


def train_xgboost(X_train, y_train, random_state=42):
    """
    Train a XGboost Classifier.

    Parameters:
    - X_train: training features.
    - y_train: training target variable.
    - random_state: int, controls the randomness of the estimator.

    Returns:
    - xgb_clf: trained XGboost Classifier.
    """
    # Fit XGBoost classifier
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', seed=42)
    xgb_clf.fit(X_train, y_train)
    return xgb_clf
