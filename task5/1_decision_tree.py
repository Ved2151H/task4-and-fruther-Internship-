"""Train a Decision Tree classifier and visualize the tree.
Outputs:
 - /mnt/data/dt_tree.png  (visualized decision tree)
 - prints train/test accuracy and classification report
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_prepare(path="D:\Subjects_Languages\Languages\ML_Internship\task5\heart.csv"):
    df = pd.read_csv(path)
    # Basic cleaning: drop rows with missing values if any
    df = df.dropna().reset_index(drop=True)
    # Assume last column is target if named 'target' or 'output' or 'HeartDisease' etc.
    if "target" in df.columns:
        y = df["target"]
        X = df.drop(columns=["target"])
    elif "output" in df.columns:
        y = df["output"]
        X = df.drop(columns=["output"])
    elif "HeartDisease" in df.columns:
        y = df["HeartDisease"]
        X = df.drop(columns=["HeartDisease"])
    else:
        # fallback: last column is target
        y = df.iloc[:, -1]
        X = df.iloc[:, :-1]
    return X, y, df

def main():
    X, y, df = load_prepare()
    X = pd.get_dummies(X, drop_first=True)  # simple encoding for categorical features
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42, test_size=0.2)
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    y_pred_test = clf.predict(X_test)
    print("Decision Tree (default) - Train acc:", accuracy_score(y_train, y_pred_train))
    print("Decision Tree (default) - Test  acc:", accuracy_score(y_test, y_pred_test))
    print("\\nClassification report (test):\\n", classification_report(y_test, y_pred_test))
    print("\\nConfusion matrix (test):\\n", confusion_matrix(y_test, y_pred_test))
    # Visualize tree
    plt.figure(figsize=(20,12))
    plot_tree(clf, feature_names=X.columns, class_names=[str(c) for c in np.unique(y.astype(str))], filled=True, fontsize=8)
    plt.title("Decision Tree (default settings)")
    out = "D:\Subjects_Languages\Languages\ML_Internship\task5"
    plt.savefig(out, bbox_inches="tight")
    print("\\nSaved decision tree plot to:", out)

if __name__ == "__main__":
    main()
