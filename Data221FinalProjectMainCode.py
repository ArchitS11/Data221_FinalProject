import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

def neural_network_model(features_train, features_test, labels_train, labels_test): #kein
    #preparing target label using one-hot encoding
    labels_train = to_categorical(labels_train)
    #this code turns integers to a binary matrix, essential for neural networks

    #neural network
    tf.random.set_seed(1) #seed to ensure model returns the same result
    neural_network = Sequential()

    #input layer
    neural_network.add(InputLayer(input_shape=(561,))) #561 features = 561 input layers

    #hidden layers
    neural_network.add(Dense(256, activation='relu')) #hidden layer 1, 256 neurons
    #data scientists use numbers like 512, 256, 128, 64, etc. for efficiency
    neural_network.add(Dropout(0.3)) #prevents overfitting by randomly deactivating 30% of neurons
    #TODO: Learn what Dropout() does
    neural_network.add(Dense(128, activation='relu'))

    #output layer
    neural_network.add(Dense(6, activation='softmax'))

    #compile model
    neural_network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #TODO: Learn what optimizer does, loss, metric; basically the params of compile()

    neural_network.fit(features_train, labels_train, epochs=20, batch_size=32)
    #TODO: Learn what batch_size does; 32 is a good number for efficiency; 128,64,32,16,8 are powers of 2

    class_probabilities = neural_network.predict(features_test)
    predicted_labels = np.argmax(class_probabilities, axis=1)
    #TODO: what does this code do? axis?
    #since we have done one-hot encoding, we convert it back to a 1D array with integers 0-6, corresponding to WALKING, STANDING, etc.

    evaluate_model("Neural Network Model", labels_test, predicted_labels)

    return neural_network#save the model


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
    # Create a new decision tree classifier with the max depth 10 (to control overfitting and underfitting) and the random state 42 (for reproducibility)
    decision_tree = DecisionTreeClassifier(max_depth=10, random_state=42, criterion="entropy")  # Make predictions using the entropy criterion

    # Train the decision tree model using the features_train and target_train
    decision_tree.fit(features_train, target_train)

    # Create a variable to store the decision tree's predictions on the testing set.
    decision_tree_model_predictions = decision_tree.predict(features_test)


    # return the evaluated results of the decision tree model
    return evaluate_model("Decision Tree Model", target_test, decision_tree_model_predictions)


def lstm_model(features_train, features_test, target_train, target_test):
    maximum_sequence_length = 200

    # Preprocessing
    features_train = pad_sequences(
        features_train,
        maxlen=maximum_sequence_length,
        padding="post",
        truncating="post"
    )

    features_test = pad_sequences(
        features_test,
        maxlen=maximum_sequence_length,
        padding="post",
        truncating="post"
    )



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
lstm_model(features_train, features_test, labels_train, labels_test)
