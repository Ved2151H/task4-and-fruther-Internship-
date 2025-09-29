"""Analyze overfitting and control tree depth.
 - Trains DecisionTrees with varying max_depth
 - Saves plot of train vs test accuracy to /mnt/data/dt_depth_vs_acc.png
 - Prints best depth by test accuracy
"""
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_prepare(path="/mnt/data/heart.csv"):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if "target" in df.columns:
        y = df["target"]; X = df.drop(columns=["target"])
    elif "output" in df.columns:
        y = df["output"]; X = df.drop(columns=["output"])
    else:
        y = df.iloc[:, -1]; X = df.iloc[:, :-1]
    return pd.get_dummies(X, drop_first=True), y

def main():
    X, y = load_prepare()
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    depths = list(range(1, 21))
    train_acc = []
    test_acc = []
    for d in depths:
        clf = DecisionTreeClassifier(max_depth=d, random_state=42)
        clf.fit(X_train, y_train)
        train_acc.append(accuracy_score(y_train, clf.predict(X_train)))
        test_acc.append(accuracy_score(y_test, clf.predict(X_test)))
    best_idx = int(np.argmax(test_acc))
    best_depth = depths[best_idx]
    print("Best max_depth by test accuracy:", best_depth, "with test acc:", test_acc[best_idx])
    plt.figure(figsize=(8,5))
    plt.plot(depths, train_acc, marker='o')
    plt.plot(depths, test_acc, marker='o')
    plt.xlabel("max_depth")
    plt.ylabel("Accuracy")
    plt.title("Decision Tree: Train vs Test Accuracy by max_depth")
    plt.legend(["Train", "Test"])
    out = "/mnt/data/dt_depth_vs_acc.png"
    plt.savefig(out, bbox_inches="tight")
    print("Saved train vs test accuracy plot to:", out)

if __name__ == "__main__":
    main()
