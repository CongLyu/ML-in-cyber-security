import os

from matplotlib.figure import Figure
from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import accuracy_score, classification_report, \
    confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler

from models.XGboost_classifier import train_xgboost
from models.decision_tree_classifier import train_decision_tree

from data_processing.feature_engineering import encode_categorical_features
from sklearn.model_selection import train_test_split

from models.neural_network_classifier import train_neural_network
from models.random_forest_classifier import train_random_forest
from models.svm_classifier import train_svm

import numpy as np
import seaborn as sns
import pandas as pd


def encode_features(df, label_encoders):
    """
    Encodes the categorical features of the dataframe using provided label encoders.
    """
    df_encoded = df.copy()
    for col, le in label_encoders.items():
        if col in df_encoded:
            df_encoded[col] = le.transform(df_encoded[col])
    return df_encoded


def predict_and_save(selected_model_name, model, X_train, X, y, label_encoder, file_path, dataset):
    """
    Uses the model to predict the response, calculates the accuracy, and saves the dataframe with predictions to a CSV.
    """
    if selected_model_name == '5 Neural Network':
        # Scale features (neural networks generally benefit from feature scaling)
        scaler = StandardScaler()
        scaler.fit_transform(X_train)
        X_scaled = scaler.transform(X)
        X = X_scaled
    # Predict the responses
    y_probs = model.predict(X)

    # Handle different model output formats, especially neural network probabilities
    if selected_model_name == '5 Neural Network':
        # Assuming output is probabilities, convert to binary predictions
        # Check if it's binary classification with probability output
        if y_probs.shape[1] > 1:
            y_pred = np.argmax(y_probs, axis=1)  # Multiclass prediction
        else:
            y_pred = (y_probs > 0.5).astype(int)  # Binary classification
    else:
        # For other models, assuming direct class outputs or already binary
        y_pred = y_probs if y_probs.ndim == 1 or y_probs.shape[1] == 1 else np.argmax(y_probs, axis=1)

    # Decode labels if label_encoder is provided and ensure y_pred is flat
    if label_encoder and y_pred.ndim == 1:
        y_pred_str = label_encoder.inverse_transform(y_pred)
        y_true_str = label_encoder.inverse_transform(y)
    else:
        y_pred_str = y_pred  # Use as is if no encoder or if y_pred is not flat
        y_true_str = y

    # Create dataframe with predictions
    df_predictions = dataset.copy()
    #pd.DataFrame(X, columns=['Feature_' + str(i) for i in range(X.shape[1])])
    df_predictions['Predicted_Response'] = y_pred_str
    #df_predictions['True_Response'] = y_true_str

    # Calculate accuracy
    accuracy = accuracy_score(y, y_pred)
    print(f'Accuracy of the model: {accuracy:.2f}')


    # Compute confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Save to CSV
    df_predictions.to_csv(file_path, index=False)
    print(f'Predictions saved to {file_path}')

    return df_predictions, accuracy, cm

class ApplicationController:
    def __init__(self, main_app):
        self.main_app = main_app  # Reference to the MainApplication instance

    def reset_shared_data(self):
        # Reset shared_data dictionary to None or default values
        self.main_app.shared_data = {
            'dataset': None,
            'dataset_reduced': None,
            'dataset_covariates': None,
            'dataset_covariates_encoded': None,
            'X_train': None,
            'X_test': None,
            'dropped_columns': [],
            'target_column': None,
            'dataset_response': None,
            'dataset_response_encoded': None,
            'Y_train': None,
            'Y_test': None,
            'Y_predicted': {},
            'selected_model': None,
            'test_size': None,
            'models': {},
            'evaluation_metrics': None,
            'label_encoder': {},
        }

    def update_shared_data(self, key, value):
        self.main_app.shared_data[key] = value

    def update_new_data(self, key, value):
        self.main_app.new_data[key] = value

    def get_shared_data(self, key):
        return self.main_app.shared_data[key]

    def get_new_data(self, key):
        return self.main_app.new_data[key]

    def show_frame(self, frame_name):
        self.main_app.show_frame(frame_name)

    def drop_dataset_column(self, columns_to_drop):
        if self.main_app.shared_data['dataset_reduced'] is None:
            self.main_app.shared_data['dataset_reduced'] = self.main_app.shared_data['dataset']
        for column_name in columns_to_drop:
            self.main_app.shared_data['dataset_reduced'] = self.main_app.shared_data['dataset_reduced'].drop(columns=column_name, errors='ignore')
        if self.main_app.shared_data['dropped_columns'] is None:
            self.main_app.shared_data['dropped_columns'] = []
        self.main_app.shared_data['dropped_columns'] = self.main_app.shared_data['dropped_columns'].extend(columns_to_drop)

    def data_loaded(self):
        self.show_frame('Frame2')

    def prepare_response_data(self):
        # Get the name of the target column and the reduced dataset from shared data
        target_column = self.get_shared_data('target_column')
        reduced_dataset = self.get_shared_data('dataset_reduced')

        # Check if both the target column and reduced dataset are available
        if target_column is not None and reduced_dataset is not None:
            # Check if the target column exists in the reduced dataset
            if target_column in reduced_dataset.columns:
                # Extract the response data as a Series
                response_series = reduced_dataset[target_column]

                # Store the response series back to shared_data for further processing
                self.update_shared_data('dataset_response', response_series)
            else:
                print(f"The target column '{target_column}' is not found in the reduced dataset.")
        else:
            print("Target column or reduced dataset is not available.")

    def encode_features_and_response(self):
        """
        Encodes the covariates and response variables and updates shared data.
        """
        # Retrieve dataset_covariates and dataset_response from shared data
        covariates = self.main_app.shared_data.get('dataset_covariates')
        response = self.main_app.shared_data.get('dataset_response')

        # Ensure 'label_encoder' is initialized as a dictionary
        if 'label_encoder' not in self.main_app.shared_data:
            self.main_app.shared_data['label_encoder'] = {}

        if covariates is not None and response is not None:
            # Encode the covariates
            covariates_encoded, label_encoders_covariates = encode_categorical_features(covariates)
            self.main_app.shared_data['dataset_covariates_encoded'] = covariates_encoded
            self.main_app.shared_data['label_encoder']['covariates'] = label_encoders_covariates

            # dataset_response also needs to be encoded (if it's categorical)
            if response.dtype == 'object':
                le_response = LabelEncoder()
                response_encoded = le_response.fit_transform(response)
                self.main_app.shared_data['dataset_response_encoded'] = response_encoded
                self.main_app.shared_data['label_encoder']['response'] = le_response
            else:
                self.main_app.shared_data['dataset_response_encoded'] = response

            # Indicate success
            return True
        else:
            # Handle the error case where covariates or response is missing
            return False

    def split_train_test_data(self):
        """
        Splits the dataset into training and test sets based on the test_size.
        """
        # Retrieve encoded covariates, response, and test_size from shared data
        covariates_encoded = self.main_app.shared_data.get('dataset_covariates_encoded')
        response_encoded = self.main_app.shared_data.get('dataset_response_encoded')
        test_size = self.main_app.shared_data.get('test_size')

        if covariates_encoded is not None and response_encoded is not None and test_size is not None:
            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(
                covariates_encoded, response_encoded, test_size=test_size, random_state=None)

            # Update the shared data
            self.main_app.shared_data['X_train'] = X_train
            self.main_app.shared_data['X_test'] = X_test
            self.main_app.shared_data['Y_train'] = y_train
            self.main_app.shared_data['Y_test'] = y_test

            # Indicate success
            return True
        else:
            # Handle the error case where covariates, response, or test_size is missing
            return False

    def train_and_predict_model(self):
        """
        Trains the selected ML model and makes predictions on the test set.
        """
        # Extract the training and test data from shared_data
        X_train = self.main_app.shared_data.get('X_train')
        y_train = self.main_app.shared_data.get('Y_train')
        X_test = self.main_app.shared_data.get('X_test')
        selected_model = self.main_app.shared_data.get('selected_model')
        # Determine if the model is a neural network
        is_neural_network = '5 Neural Network' in selected_model

        # Map the model names to the training functions
        model_functions = {
            '1 SVM': train_svm,
            '2 Decision Tree': train_decision_tree,
            '3 Random Forest': train_random_forest,
            '4 XGboost': train_xgboost,
            '5 Neural Network': train_neural_network
        }

        # Check if the data and selected model are available
        if X_train is not None and y_train is not None and X_test is not None and selected_model in model_functions:
            # Train the selected model
            train_function = model_functions[selected_model]
            model = train_function(X_train, y_train, random_state=42)

            # Ensure 'models' is initialized as a dictionary
            if 'models' not in self.main_app.shared_data:
                self.main_app.shared_data['models'] = {}

            # Store the trained model in shared_data
            self.main_app.shared_data['models'][selected_model] = model
            if is_neural_network:
                # Scale features (neural networks generally benefit from feature scaling)
                scaler = StandardScaler()
                scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                self.update_shared_data("X_test", X_test_scaled)
                X_test = X_test_scaled
            # Make predictions on the test set using the trained model
            y_predicted = model.predict(X_test)

            if is_neural_network:
                # Convert probabilities to binary outputs
                y_predicted = np.argmax(y_predicted, axis=1)

            # Ensure 'Y_predicted' is initialized as a dictionary
            if 'Y_predicted' not in self.main_app.shared_data:
                self.main_app.shared_data['Y_predicted'] = {}

            # Store the predictions in shared_data
            self.main_app.shared_data['Y_predicted'][selected_model] = y_predicted

            # Indicate success
            return True
        else:
            # If the necessary data isn't available, or selected model is not recognized
            return False

    def evaluate_model_performance(self):
        """
        Generates a confusion matrix for each model and ROC curves for all models on the same plot.
        Returns a dictionary of confusion matrix figure objects by model,
        and a single ROC curve figure with all models' curves.
        """
        Y_test = self.main_app.shared_data.get('Y_test')
        models = self.main_app.shared_data.get('models')
        Y_preds = self.main_app.shared_data.get('Y_predicted')  # This should be a dictionary of predictions

        # Prepare the figures
        cm_figures = {}
        roc_figure = Figure(figsize=(5, 4))
        ax_roc = roc_figure.add_subplot(111)

        # Check if Y_test is available
        if Y_test is not None:
            for model_name, model in models.items():
                Y_pred_classes = Y_preds.get(model_name)
                # Skip if no predictions are made for this model
                if Y_pred_classes is None:
                    continue

                # Create confusion matrix figure for the current model
                cm_figure = Figure(figsize=(5, 4))
                ax_cm = cm_figure.add_subplot(111)
                cm = confusion_matrix(Y_test, Y_pred_classes)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)  # Set annot=True to display the counts
                ax_cm.set_title(f'{model_name} Confusion Matrix')
                ax_cm.set_ylabel('True label')
                ax_cm.set_xlabel('Predicted label')
                cm_figures[model_name] = cm_figure

                # Prepare for ROC curve
                # Determine if the model is a neural network
                is_neural_network = 'Neural Network' in model_name

                if is_neural_network:
                    # Assuming the second column are the probabilities for positive class
                    Y_probs = Y_pred_classes[:, 1] if Y_pred_classes.ndim > 1 else Y_pred_classes
                elif 'SVM' in model_name:
                    Y_probs = model.decision_function(self.main_app.shared_data.get('X_test'))
                else:
                    # For models that do have predict_proba method, get probabilities for the positive class
                    Y_probs = model.predict_proba(self.main_app.shared_data.get('X_test'))[:, 1]

                fpr, tpr, _ = roc_curve(Y_test, Y_probs)
                roc_auc = auc(fpr, tpr)
                ax_roc.plot(fpr, tpr, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')

            # Customize the ROC curve plot
            ax_roc.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax_roc.set_xlim([0.0, 1.0])
            ax_roc.set_ylim([0.0, 1.05])
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic')
            ax_roc.legend(loc="lower right")

        return cm_figures, roc_figure

    # Function to read in new data and predict using the selected model
    def read_and_predict_new_data(self):
        X_train = self.main_app.shared_data.get('X_train')
        # Read the new dataset
        new_data = self.get_new_data('dataset')

        # Retrieve covariates and response column names from shared_data
        covariate_columns = self.get_shared_data('dataset_covariates').columns
        response_column = self.get_shared_data('target_column')

        # Split the new data into covariates and response based on the original data's structure
        new_covariates = new_data[covariate_columns]
        new_response = new_data[response_column]

        # Encode the new covariates using the stored label encoders
        covariate_encoders = self.get_shared_data('label_encoder')['covariates']
        new_covariates_encoded = encode_features(new_covariates, covariate_encoders)

        # Encode the new response using the stored label encoder, if it's categorical
        response_encoder = self.get_shared_data('label_encoder')['response']
        if isinstance(new_response.iloc[0], str):
            new_response_encoded = response_encoder.transform(new_response)
        else:
            new_response_encoded = new_response

        # Retrieve the selected model
        selected_model_name = self.get_shared_data('selected_model')
        model = self.get_shared_data('models')[selected_model_name]

        # Predict and save the results to a CSV file
        output_file_path = os.path.join('/Users/conglyu/6627 consult', 'predicted_output.csv')
        predictions_df, accuracy, cm = predict_and_save(selected_model_name, model, X_train, new_covariates_encoded, new_response_encoded, response_encoder, output_file_path, new_data)

        return predictions_df, accuracy, cm



    # ... any other controller methods ...
