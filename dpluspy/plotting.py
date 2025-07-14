"""
Functions for plotting data, models and probability distributions.
"""

from bokeh import palettes
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy

from . import utils


mpl.rcParams["xtick.labelsize"] = 10
mpl.rcParams["ytick.labelsize"] = 10
mpl.rcParams["legend.fontsize"] = 8
mpl.rcParams["font.size"] = 11
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["font.style"] = "normal"
mpl.rcParams["font.family"] = "sans-serif"
mpl.rcParams["savefig.bbox"] = "tight"


def plot_d_plus_curves(
    models=[],
    means=[],
    varcovs=[],
    pop_ids=None,
    bins=None,
    stats_to_plot=[],
    Hs_to_plot=[],
    labels=[],
    fill=True,
    cols=5,
    ax_size=2,
    aspect=1,
    plot_H=False,
    cM=False,
    sharey=False,
    title=None,
    ylim=None,
    grid=False,
    ylabel="$D^+$",
    hline=None,
    out=None,
    show=True,
    colors=None
):
    """
    Plot an arbitrary number of datasets and model expectations on shared axes.
    It is expected that all sets include the same populations, in the same 
    order. At minimum, you must give `bins` and either `means` and `varcovs` or
    `models`.

    Each statistic is plotted in its own panel.

    :param models: Models to plot, specified as DplusStats instances.
    :param means: One or more sets of D+ means.
    :param varcovs: One or more sets of D+ varcovs, given in the same order as
        `means`.
    :param pop_ids: Population IDs of model/data statistics. If not given 
        (default None), uses IDs from the first `models` entry. If no `models`
        were passed, uses integers.
    :param bins: Array of bin edges. Default None, but required.
    :param stats_to_plot: If given, plots only these statistics. Note that
        labels assigned to populations should match the set of `pop_ids` used.
    :param Hs_to_plot: H statistics to plot. None (default) plots all H stats
        present, if `plot_H` is True.
    :param labels: Optional list of labels for datasets and models. 
    :param fill: If True (default), plot confidence intervals as shaded regions.
        If False, makes scatter plots with error bars.
    :param cols: Number of subplot columns (default 5).
    :param ax_size: Subplot size (default 2).
    :param aspect: Ratio of subplot length/height (default 1).
    :param plot_H: If True, plot H stats in the last subplot (default False).
    :param cM: If True, plot the x axis in units of centiMorgans (default False)
    :param sharey: If True, force subplots to share a y axis scale (default
        False).
    :param title: Figure title (default None).
    :param ylim: Lower bound for y axes (default None).
    :param grid: If True, plot grids (default False).
    :param ylabel: Label for y axes (default "$D^+$").
    :param hline: Optional y-coord at which to plot a horizontal line
        (default None).
    :param out: Optional pathname of output file (default None).
    :param show: If True, display the figure with plt.show() (default).

    :returns: None
    """
    if bins is None:
        raise ValueError('You must provide `bins`')
    if len(models) == 0 and len(means) == 0:
        raise ValueError('No models or data were passed')

    # if one model is given; nest it in a list
    if type(models) != list:
        models = [models]
    # do the same with data
    if len(varcovs) != len(means):
        raise ValueError('Lengths of `means` and `varcovs` mismatch')
    if len(means) > 0:
        if type(means[0]) != list:
            means = [means]
            varcovs = [varcovs] 

    # check labels and build them if necessary
    if len(labels) == 0:
        if len(means) == 1:
            labels = ["Data"]
        else:
            labels = [f"Data {i}" for i in range(len(means))]
        if len(models) == 1:
            labels += ["Model"]
        else:
            labels += [f"Model {i}" for i in range(len(models))]

    if len(labels) != len(models) + len(means):
        raise ValueError('Label length mismatches models, data')

    # figure out pop_ids
    if pop_ids is None:
        if len(models) > 0:
            pop_ids = models[0].pop_ids
            statistics = models[0].stat_names[0]
        else:
            raise ValueError('You must provide `pop_ids`')
    else:
        statistics = utils._DP_names(pop_ids)
        if len(models) > 0:
            assert len(statistics) == len(models[0][0])
        if len(means) > 0:
            assert len(statistics) == len(means[0][0])
    
    stat_names = utils._get_latex_names(pop_ids)

    # if no stats_to_plot were given: plot all statistics
    if len(stats_to_plot) == 0:
        stats_to_plot = statistics

    num_stats = len(stats_to_plot)
    
    if plot_H:
        num_stats += 1
        if len(Hs_to_plot) == 0:
            Hs_to_plot = utils._H_names(pop_ids)
        H_labels = []
        for Hstat in Hs_to_plot:
            pop0, pop1 = Hstat.strip('H_').split('_')
            H_labels.append(f'{pop_ids.index(pop0)},{pop_ids.index(pop1)}')
        sharex = False
    else:
        sharex = True

    cols = min(cols, num_stats)
    rows = -(num_stats // -cols)
    figsize = (cols * ax_size * aspect, rows * ax_size)

    if isinstance(bins, list):
        assert len(bins) == len(models) + len(means)
        xs = [(b[1:] + b[:-1]) / 2 for b in [np.asarray(b) for b in bins]]
    else:
        bins = np.asarray(bins)
        x = (bins[1:] + bins[:-1]) / 2
        xs = [x for i in range(len(models) + len(means))]

    x_label = "$r$"
    if cM == True:
        mids = utils._map_function(mids) * 0.01
        x_label = "cM"

    fig, axs = plt.subplots(
        rows, cols, figsize=figsize, layout="constrained", 
        sharex=sharex, sharey=sharey
    )
    
    if rows > 1:
        axs = axs.flat
    elif cols == 1:
        axs = [axs]

    for ax in axs[num_stats:]:
        ax.remove()

    if colors is None:
        # special cases for color assignment
        if len(models) == 1 and len(means) == 1:
            colors = [palettes.Category10_10[0]] * 2
        elif len(labels) <= 10:
            colors = palettes.Category10_10
        else:
            colors = palettes.TolRainbow[len(labels)]
    else:
        assert len(colors) == len(models) + len(means)

    for i, stat in enumerate(stats_to_plot):
        ax = axs[i]
        j = 0
        k = statistics.index(stat)
        for mean, v in zip(means, varcovs):
            if i == 0:
                label = labels[j]
            else:
                label = None
            x = xs[j]
            err = np.array([v[l][k, k] ** 0.5 * 1.96 for l in range(len(x))])
            y = np.array([mean[j][k] for j in range(len(x))])
            if fill:
                ax.fill_between(x, y - err, y + err, alpha=0.3, color=colors[j],
                                edgecolor='none')
                ax.plot(x, y, linestyle="dashed", color=colors[j], label=label)
            else:
                ax.errorbar(
                    x, y, yerr=err, fmt='o', mfc='none', mec=colors[j],
                    label=labels[j]
                )
            j += 1
        for model in models:
            if i == 0:
                label = labels[j]
            else:
                label = None
            x = xs[j]
            y = [model[l][k] for l in range(len(x))]
            ax.plot(x, y, color=colors[j], label=label)
            j += 1
        # ax setup
        if hline:
            ax.hlines(hline, bins[0], bins[-1], colors='black')
        ax.set_xscale("log")
        if i >= num_stats - cols:
            ax.set_xlabel(x_label)
        if i % cols == 0:
            ax.set_ylabel(ylabel)
        stat_name = stat_names[k]
        ax.set_title(stat_name, y=0.85)
        if ylim is not None:
            ax.set_ylim(ylim,)
        if grid:
            ax.grid(alpha=0.3)

    if plot_H:
        ax = axs[num_stats - 1]
        for k, label in enumerate(Hs_to_plot):
            j = 0
            for mean, v in zip(means, varcovs):
                if k == 0:
                    label = labels[j]
                else:
                    label = None
                err = [v[-1][k, k] ** 0.5 * 1.96]
                y = [mean[-1][k]]
                ax.errorbar(
                    k, y, yerr=err, fmt='o', mfc='none', mec=colors[j],
                    ecolor=colors[j]
                )
                j += 1
            for model in models:    
                if k == 0:
                    label = labels[j]
                else:
                    label = None
                y = [model[-1][k]]
                ax.scatter(k, y, color=colors[j], marker='x')
                j += 1
        ax.set_xticks(list(range(len(Hs_to_plot))), labels=H_labels)
        ax.set_xlim(-0.2, len(Hs_to_plot) - 0.8)
        ax.set_title('$H$', y=0.85)
        if ylim is not None:
            ax.set_ylim(ylim,)
        if grid:
            ax.grid(alpha=0.3)

    if len(labels) > 1:
            ncols = int(cols * ax_size)
            fig.legend(
                framealpha=0, loc='lower center', ncols=ncols, 
                bbox_to_anchor=(0.5, -0.1))
    if title:
        fig.suptitle(title, x=0.04, horizontalalignment='left')
    else:
        fig.suptitle("")

    if out:
        if out.endswith(".pdf"):
            plt.savefig(out, bbox_inches='tight', pad_inches=0.1)
        else:
            plt.savefig(out, dpi=244, bbox_inches='tight', pad_inches=0.1)
    if show:
        plt.show()

    return 


def plot_parameters(
    pnames, 
    params,
    lls=None,
    params_to_plot=None,
    ax_size=1.5,
    out=None,
    show=True
):
    """
    Make a scatter plot of fitted parameters.

    :param pnames: List of parameter names.
    :param params: Array of fitted parameters from multiple replicates. Shape
        should be (number of replicates, number of parameters)
    :param list lls: Optional list of log-likelihoods, used to color points.
    :param params_to_plot: Optional list of parameters in `pnames` to plot.
        (default None).
    :param ax_size: Subplot size (default 1.5).
    :param out: Optional path at which to save the figure.
    :param show: If True (default), display the figure with plt.show().
    :returns: None
    """
    assert len(pnames) == params.shape[1]
    if params_to_plot is not None: 
        idx = [pnames.index(param) for param in params_to_plot]
        params = params[:, idx]
        pnames = params_to_plot
    num_stats = len(pnames)
    figsize = (num_stats * ax_size, num_stats * ax_size)
    fig, axs = plt.subplots(num_stats, num_stats, figsize=figsize, 
        sharey='row', sharex='col', layout='constrained')
    for i, name_i in enumerate(pnames):
        for j, name_j in enumerate(pnames):
            ax = axs[i, j]
            ax.ticklabel_format(useOffset=False)
            if j == i:
                ax.annotate(name_i, (0.3, 0.5), xycoords='axes fraction',
                            fontsize=9)
            else:
                if lls is None:
                    ax.scatter(params[:, j], params[:, i], marker='o', 
                        c='none', edgecolors='black')
                else:
                    #ax.scatter(params[:, j], params[:, i], marker="o",
                    #    c="none", edgecolors=colors)
                    cmap = ax.scatter(params[:, j], params[:, i], marker="x",
                        c=lls, cmap="viridis")
            if i == num_stats - 1:
                ax.tick_params(axis='x', labelrotation=90)
    if lls is not None:
        fig.colorbar(cmap, ax=axs[-1, -1])
    if out:
        plt.savefig(out, dpi=244, bbox_inches='tight')
    if show:
        plt.show()
    return


def plot_gamma_pdf(shape, scale):
    """
    Plot a PDF of the gamma distribution with parameters `shape` and `scale`.
    """
    fig, ax = plt.subplots(layout='constrained')
    cutoff = scipy.stats.gamma.ppf(1-1e-5, shape, scale=scale)
    x = np.linspace(0, cutoff, 1000)
    y = scipy.stats.gamma.pdf(x, shape, scale=scale)
    ax.plot(x, y)
    plt.show()
    return 

