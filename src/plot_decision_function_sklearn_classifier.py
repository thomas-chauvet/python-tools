import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import scoreatpercentile

def plot_decision_function(clf, X, y):
    """
    Plot decision function of a binary classifier in two dimensions.
    :param clf: binary classifier using sklearn API convention (has a `decision_function` method).
    :param X: features set (matrix pandas DataFrame or numpy 2D array).
    :param y: target (vector pandas Series or numpy 1D array).
    :return: a plot with data and decision boundaries.
    """

    xx, yy = np.meshgrid(
        np.linspace(np.floor(np.min(X[:, 0])), np.ceil(np.max(X[:, 0])), 100),
        np.linspace(np.floor(np.min(X[:, 1])), np.ceil(np.max(X[:, 1])), 100))

    scores_pred = clf.decision_function(X) * -1

    threshold = scoreatpercentile(scores_pred, 100 * y.sum() / len(y))

    z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
    z = z.reshape(xx.shape)

    plt.contourf(
        xx, yy, z, levels=np.linspace(z.min(), 0.1, 7), cmap=plt.cm.Blues_r)
    plt.contour(xx, yy, z, levels=[threshold], linewidths=2, colors='red')
    plt.contourf(xx, yy, z, levels=[threshold, z.max()], colors='orange')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="white")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="black")
    plt.show()