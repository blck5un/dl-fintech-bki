import matplotlib.pyplot as plt


def plot_losses(
    df,
    figsize=(8, 12),
    loss_lims=None,
    metirc_lims=None,
    save_to_file=None,
):
    fig, axs = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=figsize,
        sharex=True,
        tight_layout=True,
        gridspec_kw={'height_ratios': [2, 2, 1]}
    )
    cmap = plt.get_cmap('tab10')
    xs = df['epoch']
    metrics = ['loss', 'roc_auc']
    modes = ['train', 'valid']
    for i, metric in enumerate(metrics):
        for mode in modes:
            col = f'{mode}_{metric}'
            axs[i].plot(xs, df[col], label=mode.capitalize(), marker='.')
    col = 'lr'
    axs[2].plot(xs, df[col], drawstyle='steps-post', label=col, marker='.', color=cmap(2))
    titles = ['Loss', 'ROC_AUC', 'Learning rate']
    ylabels = ['loss', 'roc_auc', 'lr']
    for i, ax in enumerate(axs):
        ax.set_title(titles[i], fontsize=16)
        ax.set_ylabel(ylabels[i], fontsize=14)
        ax.legend(fontsize=14)
        ax.grid(linestyle='--')
        ax.tick_params(axis='both', which='major', labelsize=12)
    if loss_lims is not None:
        axs[0].set_ylim(*loss_lims)
    if metirc_lims is not None:
        axs[1].set_ylim(*metirc_lims)
    axs[2].set_xlabel('epoch', fontsize=14)
    axs[2].set_yscale('log')
    axs[2].set_xlim(0, xs.max() + 1)
    if save_to_file is not None:
        fig.savefig(save_to_file)
