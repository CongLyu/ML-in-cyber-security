from sklearn.preprocessing import LabelEncoder


def drop_columns(dataframe, columns_to_drop):
    """
    Drop specified columns from a DataFrame.

    Parameters:
    - dataframe: pandas DataFrame from which the columns will be dropped.
    - columns_to_drop: list of strings representing the column names to be dropped.

    Returns:
    - The DataFrame after dropping the specified columns.
    """
    # Replicate the original DataFrame to avoid modifying it in-place
    dropped_df = dataframe.copy()

    # Drop the specified columns
    dropped_df = dropped_df.drop(columns=columns_to_drop, errors='ignore')

    return dropped_df


def encode_categorical_features(features):
    """
    Encode categorical features to numeric.

    Parameters:
    - features: pandas DataFrame with features to encode.

    Returns:
    - features_encoded: DataFrame with encoded features.
    - label_encoders: Dictionary of LabelEncoder objects for each categorical column.
    """
    label_encoders = {}
    for col in features.columns:
        if features[col].dtype == 'object':
            le = LabelEncoder()
            features[col] = le.fit_transform(features[col])
            label_encoders[col] = le
    return features, label_encoders


def select_prediction_column(dataframe, target_column):
    """
    Selects the column to be used as the prediction target.

    Parameters:
    - dataframe: pandas DataFrame that contains the data.
    - target_column: str, the name of the column to use as the target.

    Returns:
    - target: Series containing the target data.
    - features: DataFrame containing the features with the target column dropped.
    """
    target = dataframe[target_column]
    features = dataframe.drop(columns=[target_column])
    return target, features
