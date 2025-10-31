import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def plot_svm(
    clf, X, y,
    X_test=None, y_test=None,
    padding=0.15, mesh_steps=400,
    show_margins=True, show_support=True,
    title_prefix="SVM"
):
    # ----- bounds with padding
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    xr = x_max - x_min; yr = y_max - y_min
    x_min, x_max = x_min - padding*xr, x_max + padding*xr
    y_min, y_max = y_min - padding*yr, y_max + padding*yr

    # ----- mesh
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, mesh_steps),
        np.linspace(y_min, y_max, mesh_steps)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # ----- predictions over grid (handles OvR multiclass)
    Z = clf.predict(grid).reshape(xx.shape)

    # optional: "confidence" via max distance to hyperplane(s)
    if hasattr(clf, "decision_function"):
        df = clf.decision_function(grid)
        if df.ndim == 1:                # binary
            conf = np.abs(df)
        else:                            # multiclass OvR
            conf = np.max(df, axis=1)
        conf = conf.reshape(xx.shape)
    else:
        conf = None

    # ----- base figure
    plt.figure(figsize=(7.2, 5.8), dpi=120)

    # class region colors (avoid specifying exact colors if you prefer defaults)
    cmap_regions = ListedColormap(plt.cm.Pastel1.colors[:len(np.unique(y))])

    # decision regions
    plt.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_regions)

    # decision boundary and margins (binary only)
    if show_margins and hasattr(clf, "decision_function"):
        dvals = clf.decision_function(grid)
        if dvals.ndim == 1:  # binary case
            dvals = dvals.reshape(xx.shape)
            # 0 = boundary; ±1 = margins
            cs = plt.contour(xx, yy, dvals, levels=[-1, 0, 1],
                             linestyles=['--', '-', '--'],
                             linewidths=[1.2, 1.8, 1.2], colors='k')
            #cs.collections[1].set_label("Decision boundary")

    # training points
    plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k', linewidths=0.6, alpha=0.9)

    # test points (optional)
    if X_test is not None and y_test is not None:
        y_pred_test = clf.predict(X_test)
        correct = y_pred_test == y_test
        # correct: hollow squares; incorrect: x markers
        plt.scatter(X_test[correct,0], X_test[correct,1],
                    facecolors='none', edgecolors='k', marker='s', s=70, linewidths=1.2,
                    label='Test (correct)')
        if np.any(~correct):
            plt.scatter(X_test[~correct,0], X_test[~correct,1],
                        marker='x', s=60, linewidths=1.8, label='Test (misclassified)')

    # support vectors
    if show_support and hasattr(clf, "support_vectors_"):
        sv = clf.support_vectors_
        plt.scatter(sv[:,0], sv[:,1],
                    s=120, facecolors='none', edgecolors='k', linewidths=1.5,
                    label='Support vectors')

    # cosmetics
    plt.xlim(x_min, x_max); plt.ylim(y_min, y_max)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
    ttl = f"{title_prefix}: Decision Boundary"
    if X_test is not None and y_test is not None:
        acc = (clf.predict(X_test) == y_test).mean()
        ttl += f"  |  Test acc: {acc:.3f}"
    plt.title(ttl)
    # deduplicate legend labels
    handles, labels = plt.gca().get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    if uniq:
        plt.legend(uniq.values(), uniq.keys(), frameon=False, loc='best')
    plt.tight_layout()
    plt.show()