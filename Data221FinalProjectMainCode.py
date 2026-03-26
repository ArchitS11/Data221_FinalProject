import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.layers import Dense, InputLayer, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

def load_and_prepare_data(train_path='train.csv', test_path='test.csv'):
    # Load the kaggle datasets
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Combine them into one large dataset (10,299 instances)
    full_data = pd.concat([train_df, test_df], ignore_index=True)

    # Separate features (X) and target (y)
    # We drop 'subject' because it is an identifier, not a sensor feature
    X = full_data.drop(columns=['Activity', 'subject'])
    y = full_data['Activity']

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Perform the 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,stratify=y, random_state=42)

    # Initialize and apply StandardScaler to prevent data leakage
    scaler = StandardScaler()

    # Fit ONLY on the training data, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

def run_knn(X_train, X_test, y_train, y_test, k=5):
    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    evaluate_model(knn, y_test, y_pred)

    return knn

def neural_network_model(X_train, X_test, y_train, y_test): #kein
    #preparing target label using one-hot encoding
    y_train = to_categorical(y_train)
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

    neural_network.fit(X_train, y_train, epochs=20, batch_size=32)
    #TODO: Learn what batch_size does; 32 is a good number for efficiency; 128,64,32,16,8 are powers of 2

    class_probabilities = neural_network.predict(X_test)
    y_pred = np.argmax(class_probabilities, axis=1)
    #TODO: what does this code do? axis?
    #since we have done one-hot encoding, we convert it back to a 1D array with integers 0-6, corresponding to WALKING, STANDING, etc.

    evaluate_model(neural_network, y_test, y_pred)

    return neural_network#save the model


def evaluate_model(model_name, y_true, y_pred):

    # Calculate metrics (using 'weighted' because we have 6 activity classes, not just 2)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted') #using average='weighted' because we have 6 classes, not 2
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    confusion = confusion_matrix(y_true, y_pred)

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
    decisionTreeModel = DecisionTreeClassifier(max_depth=10, random_state=42, criterion="entropy")  # Make predictions using the entropy criterion

    # Train the decision tree model using the features_train and target_train
    decisionTreeModel.fit(features_train, target_train)

    # Create a variable to store the decision tree's predictions on the testing set.
    decisionTreeModelPredictions = decisionTreeModel.predict(features_test)

    # return the evaluated results of the decision tree model
    return evaluate_model("Decision Tree Model", target_test, decisionTreeModelPredictions)

# -----------------------------
# Main program
# -----------------------------

#TODO: We don't need to put Main Program into our own repository, just the functions of our model
X_train, X_test, y_train, y_test = load_and_prepare_data()

run_knn(X_train, X_test, y_train, y_test)
neural_network_model(X_train, X_test, y_train, y_test)
decision_tree_model(X_train, X_test, y_train, y_test)


