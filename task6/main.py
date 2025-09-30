from data_preprocessing import load_and_preprocess_data
from knn_model import train_knn
from evaluation import evaluate_model
from visualization import plot_decision_boundaries
import pandas as pd

def main():
    # Load and preprocess
    X_train, X_test, y_train, y_test = load_and_preprocess_data("Iris.csv")

    # Train model
    model = train_knn(X_train, y_train, k=5)

    # Evaluate
    evaluate_model(model, X_test, y_test)

    # Visualize
    # Combine X and y for visualization
    import numpy as np
    X_all = np.vstack((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    plot_decision_boundaries(model, X_all, y_all)

if __name__ == "__main__":
    main()
