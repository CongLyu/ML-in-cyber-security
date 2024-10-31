import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical


def train_neural_network(X_train, y_train, random_state):
    # Scale features (neural networks generally benefit from feature scaling)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Assuming y_train is not already one-hot encoded. Adjust if it's already one-hot encoded.
    y_train_encoded = to_categorical(y_train)

    # Neural network architecture
    nn_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dense(64, activation='relu'),
        Dense(y_train_encoded.shape[1], activation='softmax')  # Adjust the number of neurons to the number of classes
    ])

    # Compile the model
    nn_model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',  # Use 'binary_crossentropy' if it's a binary classification
        metrics=['accuracy']
    )

    # Fit the model
    nn_model.fit(
        X_train_scaled,
        y_train_encoded,
        epochs=100,
        batch_size=32,
        verbose=1
    )

    # Return the trained model
    return nn_model

# Example usage:
# model = train_neural_network(X_train, y_train)
# To evaluate or use the model, you can now call model.evaluate or model.predict
