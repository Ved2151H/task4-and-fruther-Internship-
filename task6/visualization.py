import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

def plot_decision_boundaries(model, X, y):
    # Reduce to 2D using PCA
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # Predict on grid
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    # Plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    scatter = plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=pd.factorize(y)[0], edgecolor="k", cmap="viridis")
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("KNN Decision Boundaries")
    plt.legend(*scatter.legend_elements(), title="Classes")
    plt.show()
