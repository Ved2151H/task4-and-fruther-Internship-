import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_data(file_path="Iris.csv"):
    # Load dataset
    data = pd.read_csv(file_path)

    # Drop Id column if present
    if "Id" in data.columns:
        data = data.drop(columns=["Id"])

    # Features and labels
    X = data.drop(columns=["Species"]).values
    y = data["Species"].values

    # Normalize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return X_train, X_test, y_train, y_test
