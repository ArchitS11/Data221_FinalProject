import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.layers import Dense, InputLayer, Dropout, LSTM, BatchNormalization



def load_and_prepare_data(train_path='train.csv', test_path='test.csv'):
    # Load the kaggle datasets
    train_data_frame = pd.read_csv(train_path)
    test_data_frame = pd.read_csv(test_path)

    # Combine them into one large dataset (10,299 instances)
    full_data = pd.concat([train_data_frame, test_data_frame], ignore_index=True)

    # Separate features (X) and target (y)
    # We drop 'subject' because it is an identifier, not a sensor feature
    feature_matrix = full_data.drop(columns=['Activity', 'subject'])
    target_label = full_data['Activity']

    label_encoder = LabelEncoder()
    target_label = label_encoder.fit_transform(target_label)

    # Perform the 80/20 stratified split
    features_train, features_test, labels_train, labels_test = train_test_split(feature_matrix, target_label,test_size=0.20,stratify=target_label, random_state=42)

    # Initialize and apply StandardScaler to prevent data leakage
    scaler = StandardScaler()

    # Fit ONLY on the training data, then transform both
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    return features_train_scaled, features_test_scaled, labels_train, labels_test

def knn_model(features_train, features_test, labels_train, labels_test, k=3):
    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(features_train, labels_train)

    # Make predictions
    predicted_labels = knn.predict(features_test)

    evaluate_model("K-Nearest Neighbors (KNN) Model", labels_test, predicted_labels)

    return knn #returns the model (saves the brain)


def neural_network_model(features_train, features_test, labels_train, labels_test):  # kein
    #With assistance from Gemini to come up with implementation strategy and SciKeras
    # preparing target label using one-hot encoding
    labels_train_encoded = to_categorical(labels_train)

    # this code turns integers to a binary matrix, essential for neural networks

    #nested function to build a neural network model with default sizes 256 hidden layer 1 and 128 hidden layer 2
    def build_neural_model(layer_1_size=256, layer_2_size=128):
        tf.random.set_seed(1)  # seed to ensure model returns the same result
        neural_model = Sequential()

        # input layer
        neural_model.add(InputLayer(shape=(561,)))  # 561 features = 561 input layers

        # hidden layers (Sizes are now completely tunable!)
        neural_model.add(Dense(layer_1_size, activation='relu'))
        neural_model.add(Dense(layer_2_size, activation='relu'))

        # output layer
        neural_model.add(Dense(6, activation='softmax'))

        # compile model
        neural_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return neural_model

    keras_estimator = KerasClassifier( #calls nested blueprint above
        model=build_neural_model , layer_1_size=256, layer_2_size=128, epochs=15, batch_size=32
    )

    param_grid = { #the test hidden layer sizes for GridCV
        'layer_1_size': [128, 256],
        'layer_2_size': [64, 128]
    }

    print("Cross-validation using GridSearchCV for Neural Networks: ")
    grid = GridSearchCV(estimator=keras_estimator, param_grid=param_grid, cv=3, scoring='accuracy') #cross validation with 3 folds

    grid.fit(features_train, labels_train_encoded)

    print(f"The best hidden layer sizes were: {grid.best_params_}")

    best_neural_model = grid.best_estimator_.model_ #get model

    # predicting using the best neural model
    class_probabilities = best_neural_model.predict(features_test)
    predicted_labels = np.argmax(class_probabilities, axis=1)
    # since we have done one-hot encoding, we convert it back to a 1D array with integers 0-5

    # Evaluate the best neural model
    evaluate_model("Neural Network", labels_test, predicted_labels)

    return best_neural_model  # save the smartest model


def evaluate_model(model_name, true_labels, predicted_labels):

    # Calculate metrics (using 'weighted' because we have 6 activity classes, not just 2)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted') #using average='weighted' because we have 6 classes, not 2
    recall = recall_score(true_labels, predicted_labels, average='weighted')
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    confusion = confusion_matrix(true_labels, predicted_labels)

    # Print the results cleanly
    print(f"{model_name} Performance")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", confusion)
    print("-" * 35 + "\n")


def decision_tree_model(features_train, features_test, target_train, target_test):
    # Create a new decision tree classifier with the max depth 10 (to control overfitting and underfitting)
    # Note: max_depth of 10 is optimal from rigorous experimenting on the model separately.
    # and the random state 42 (for reproducibility)
    #  Make predictions using the entropy criterion.
    decisionTreeModel = DecisionTreeClassifier(max_depth=10, random_state=42, criterion="entropy")

    # Train the decision tree model using the features_train and target_train
    decisionTreeModel.fit(features_train, target_train)

    # Create a variable to store the decision tree's predictions on the testing set.
    decision_tree_model_predictions = decisionTreeModel.predict(features_test)

    # Evaluate the model using the evaluate_model function
    evaluate_model("Decision Tree Model", target_test, decision_tree_model_predictions)

    # return the evaluated results of the decision tree model
    return decisionTreeModel

# Bonus Model
def lstm_model(features_train, features_test, target_train, target_test):

   # Set random seeds for reproducibility
    tf.random.set_seed(42)
    np.random.seed(42)

    # Reshape the Data
    # Treat each of the 561 sensor readings as 1 timestep with 1 feature value.
    features_train_lstm = features_train.reshape(features_train.shape[0], features_train.shape[1], 1)
    features_test_lstm = features_test.reshape(features_test.shape[0], features_test.shape[1], 1)

    # Build the LSTM model
    lstmModel = Sequential([

        # Input shape: (561 timesteps, 1 feature per timestep)
        tf.keras.layers.Input(shape=(features_train_lstm.shape[1], 1)),

        # First LSTM layer, return_sequences=True passes the full sequence
        LSTM(32, return_sequences=True),

        # Normalise activations between layers to stabilise and speed up training
        BatchNormalization(),

        # Second LSTM layer return_sequences=False (default) because
        # we only need the final output for classification
        LSTM(16, return_sequences=False),

        BatchNormalization(),

        # Fully connected layer to learn higher-level combinations of LSTM outputs
        Dense(32, activation="relu"),

        # Dropout randomly disables 30% of neurons during training to prevent the model from overfitting to training data
        Dropout(0.3),

        # Output layer — 6 neurons (one per activity class)
        # Softmax converts raw scores into probabilities that sum to 1
        Dense(6, activation="softmax")
    ])

    # Compile the LSTM model
    lstmModel.compile(
        # Adam adapts the learning rate automatically — good general-purpose optimiser
        optimizer="adam",

        # use sparse_categorical_crossentropy because our labels
        # are integers (0–5), not one-hot encoded vectors
        loss="sparse_categorical_crossentropy",

        metrics=["accuracy"]
    )

    # Train the LSTM model
    lstmModel.fit(
        features_train_lstm,
        target_train,
        epochs=20,
        batch_size=64,  # Feed 64 samples at a time
        verbose=0
    )

    # Get predicted class indices (argmax picks the highest probability class)
    lstmModel_predictions = np.argmax(lstmModel.predict(features_test_lstm), axis=1)

    # Use the shared evaluate_model function for consistent reporting
    evaluate_model("LSTM Model", target_test, lstmModel_predictions)
    return lstmModel


def logistic_regression_model(features_train, features_test, labels_train, labels_test):
    # Create logistic regression model
    logistic_regression = LogisticRegression(max_iter=2000, random_state=42)

    # Train the model on the training data
    logistic_regression.fit(features_train, labels_train)

    # Predict the activity labels for the test data
    predicted_labels = logistic_regression.predict(features_test)

    # Evaluate the model using shared evaluation function
    evaluate_model("Logistic Regression Model", labels_test, predicted_labels)

    return logistic_regression

# -----------------------------
# Main program
# -----------------------------

features_train, features_test, labels_train, labels_test = load_and_prepare_data()

#knn_model(features_train, features_test, labels_train, labels_test)
#neural_network_model(features_train, features_test, labels_train, labels_test)
#decision_tree_model(features_train, features_test, labels_train, labels_test)
#logistic_regression_model(features_train, features_test, labels_train, labels_test)
#lstm_model(features_train, features_test, labels_train, labels_test)

