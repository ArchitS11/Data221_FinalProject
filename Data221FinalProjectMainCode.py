import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


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
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=label_encoder.classes_)
    matrix = confusion_matrix(y_test, y_pred)

    return accuracy, report, matrix, y_pred

# -----------------------------
# Main program
# -----------------------------
X_train_scaled, X_test_scaled, y_train, y_test, label_encoder = load_and_prepare_data()

print("Preprocessing complete.")
print("X_train shape:", X_train_scaled.shape)
print("X_test shape:", X_test_scaled.shape)

accuracy, report, matrix, y_pred = run_knn(
    X_train_scaled,
    X_test_scaled,
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

