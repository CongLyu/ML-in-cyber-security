from models.model_trainer import split_data
from data_processing.feature_engineering import drop_columns, \
    encode_categorical_features, select_prediction_column
from data_processing.data_loader import load_dataset
from models.decision_tree_classifier import train_decision_tree
from models.performance_evaluation import evaluate_and_plot_decision_tree


# Specify the path to your XML file
xml_file_path = r'/Users/conglyu/6627 consult/Source Code Old Version START HERE/Sample Code and Input/Additional Datasets and Code (Code Dx)/atom (30 May 2020).xml'

# Load the dataset
findings_df = load_dataset(xml_file_path)

# Proceed with using findings_df for data processing
# Feature engineering steps
# Define columns to drop
columns_to_drop = ['cwe-href', 'rule-code', 'first-seen', 'last-updated']

# Drop the columns in the set
findings_df = drop_columns(findings_df, columns_to_drop)

# Feature engineering
columns_to_exclude = ['id']
features = drop_columns(findings_df, columns_to_exclude)
features, label_encoders = encode_categorical_features(features)

# Prepare the target variable 'status'
target = label_encoders['status'].transform(findings_df['status'])

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = split_data(features, target, test_size=0.8, random_state=42)

# Train the Decision Tree Classifier
clf = train_decision_tree(X_train, y_train)

# Evaluate the classifier and plot the decision tree
plot_path = evaluate_and_plot_decision_tree(
    clf,
    X_test,
    y_test,
    feature_names=features.columns.tolist(),
    class_names=label_encoders['status'].classes_.tolist()
)


