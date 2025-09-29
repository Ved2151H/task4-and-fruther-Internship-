"""Interpret feature importances from a Random Forest and present a simple explanation.
 - Prints sorted importances to console and saves CSV to /mnt/data/feature_importances.csv
"""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def load_prepare(path="heart.csv"):
    df = pd.read_csv(path).dropna().reset_index(drop=True)
    if "target" in df.columns:
        y = df["target"]; X = df.drop(columns=["target"])
    elif "output" in df.columns:
        y = df["output"]; X = df.drop(columns=["output"])
    else:
        y = df.iloc[:, -1]; X = df.iloc[:, :-1]
    return pd.get_dummies(X, drop_first=True), y, df

def main():
    X, y, df = load_prepare()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)

    # Feature importances
    importances = rf.feature_importances_
    fi = pd.DataFrame({"feature": X.columns, "importance": importances})
    fi = fi.sort_values("importance", ascending=False).reset_index(drop=True)

    print("Top features by importance:")
    print(fi.head(15).to_string(index=False))

    fi.to_csv("/mnt/data/feature_importances.csv", index=False)

    # Save interpretation text
    with open("/mnt/data/feature_importance_interpretation.txt", "w") as f:
        f.write("Feature importance interpretation (top 10):\n")
        for i, row in fi.head(10).iterrows():
            f.write(f"{i+1}. {row['feature']}: importance={row['importance']:.4f}\n")
        f.write(
            "\nNotes:\n"
            "- Higher importance means the model used the feature more for splitting decisions.\n"
            "- Correlated features can split importance among them.\n"
            "- Use partial dependence or SHAP for deeper causal interpretation.\n"
        )


    

if __name__ == "__main__":
    main()
