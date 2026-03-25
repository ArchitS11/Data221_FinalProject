import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
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
    y_encoded = label_encoder.fit_transform(y)

    # Perform the 80/20 stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,stratify=y, random_state=42)

    # Initialize and apply StandardScaler to prevent data leakage
    scaler = StandardScaler()

    # Fit ONLY on the training data, then transform both
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, label_encoder

def run_knn(X_train, X_test, y_train, y_test, label_encoder, k=5):
    # Create KNN model
    knn = KNeighborsClassifier(n_neighbors=k)

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions
    y_pred = knn.predict(X_test)

    # Calculate evaluation metrics
    #TODO: we're gonna make accuracy, confusion matrix, etc. to another function called evaluate_model()
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix, y_pred

def knn_model(X_train, X_test, y_train, y_test): #kein
    #preparing target label using one-hot encoding
    y_train = to_categorical(y_train)
    #this code turns integers to a binary matrix, essential for neural networks

    #neural network
    tf.random_set_seed(1) #seed to ensure model returns the same result
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

    #TODO: evaluate_model(y_test, y_pred)

    return neural_network #save the model

# -----------------------------
# Main program
# -----------------------------

#TODO: We don't need to put Main Program into our own repository, just the functions of our model
X_train, X_test, y_train, y_test, label_encoder = load_and_prepare_data()


accuracy, report, matrix, y_pred = run_knn(
    X_train,
    X_test,
    y_train,
    y_test,
    label_encoder,
    k=5
)

print("\nKNN Results")
print("Accuracy:", accuracy)

print("\nClassification Report:")
print(report)

print("\nConfusion Matrix:")
print(matrix)

