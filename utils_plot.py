import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_prediction_scatter(y_test, y_pred):
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


def plot_prediction_ts(test_dates, final_predictions, test_labels):
    df_to_compare = pd.DataFrame({'date': test_dates, 'Actual': test_labels, 'Predicted': final_predictions})
    dfm = pd.melt(df_to_compare, id_vars=['date'], value_vars=['Actual', 'Predicted'], var_name='data', value_name='precip')
    f, axs = plt.subplots(1,2,
                      figsize=(12,5),
                      sharey=True)

    sns.regplot(data= df_to_compare,
                x="Actual",
                y="Predicted",
                ax=axs[0],
                )

    sns.lineplot(x='date', y='precip', hue = 'data', data=dfm, ax=axs[1])