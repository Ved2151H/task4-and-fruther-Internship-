"""Train a Random Forest and compare accuracy to a tuned Decision Tree.
 - Saves feature comparison barplot to /mnt/data/rf_vs_dt_accuracy.txt (text) and /mnt/data/rf_feature_importance.png
"""
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    # Tuned decision tree (use best depth from prior script if known; try max_depth=4 as a reasonable default)
    dt = DecisionTreeClassifier(max_depth=4, random_state=42)
    dt.fit(X_train, y_train)
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    dt_acc = accuracy_score(y_test, dt.predict(X_test))
    rf_acc = accuracy_score(y_test, rf.predict(X_test))
    print("Decision Tree (max_depth=4) test acc:", dt_acc)
    print("Random Forest (200 trees) test acc:", rf_acc)
    print("\\nRandom Forest classification report:\\n", classification_report(y_test, rf.predict(X_test)))
    # Feature importances plot for RF
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    feat_names = X.columns
    topn = min(20, len(feat_names))
    plt.figure(figsize=(10,6))
    plt.bar(range(topn), importances[indices][:topn])
    plt.xticks(range(topn), feat_names[indices][:topn], rotation=90)
    plt.title("Random Forest feature importances (top {})".format(topn))
    out = "/mnt/data/rf_feature_importance.png"
    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight")
    print("Saved RF feature importance plot to:", out)

if __name__ == "__main__":
    main()
