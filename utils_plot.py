import matplotlib.pyplot as plt


def plot_scatter(y_test, y_pred):
    n_cols = 3
    n_plots = y_pred.shape[1]
    nrows = -(-n_plots // n_cols)

    fig, axs = plt.subplots(nrows=nrows, ncols=n_cols,
                            sharex=False, figsize=(17, 11))
    for i in range(n_plots):
        ax = plt.subplot(nrows, n_cols, i + 1)
        x = y_test.iloc[:, i]
        y = y_pred[:, i]
        ax.scatter(x, y, facecolors='none', edgecolors='k', alpha=0.5)
        v_max = max(x.max(), y.max())
        ax.plot([0, v_max], [0, v_max], 'r--')
        ax.set_xlabel('Observed prec. [mm]')
        ax.set_ylabel('Predicted prec. [mm]')
        ax.set_title(y_test.iloc[:, i].name)
        ax.set_xlim([x.min(), 1.05 * v_max])
        ax.set_ylim([y.min(), 1.05 * v_max])