
from bokeh import palettes
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import stats

from . import parsing, inference, utils


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
    rows=None,
    cols=None,
    ax_size=2,
    aspect=1,
    plot_H=False,
    cM=False,
    out=None,
    show=True,
    sharey=False,
    title=None,
    ylim=None,
    grid=False,
    ylabel=None,
    hline=None
):
    """
    Plot any number of model expectations and empirical data sets alongside
    each other. It is expected that all sets have the same number of populations
    and the same configuration of bins.

    :param models: One or more models to plot (default None). Models should be
        specified as DplusStats instances.
    :type models: list, optional
    :param means: One or more sets of D+ means.
    :type means: list, optional
    :type varcovs: One or more sets of D+ varcovs, given in the same order as
        `means`.
    :type varcovs: list, optional
    :param pop_ids: Population IDs of model/data statistics. If not given 
        (default None), uses IDs from the first `models` entry. If no `models`
        were passed, uses integers.
    :type pop_ids: list, optional
    :param stats_to_plot: If given, plots only these statistics. Note that
        labels assigned to populations should match whatever set of `pop_ids`
        is being used, as discussed above.
    :type stats_to_plot: list of str
    :param labels: Optional list of labels for models, data, in that order. 

    :returns: None
    """
    if ylabel is None: 
        ylabel = "$D^+$" 

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
        if len(models) == 1:
            labels = ["Model"]
        else:
            labels = [f"Model {i}" for i in range(len(models))]
        if len(means) == 1:
            labels += ["Data"]
        else:
            labels += [f"Data {i}" for i in range(len(means))]

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
        statistics = utils._get_Dplus_names(pop_ids)
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
            Hs_to_plot = utils._get_H_names(pop_ids)
        H_labels = []
        for Hstat in Hs_to_plot:
            pop0, pop1 = Hstat.strip('H_').split('_')
            H_labels.append(f'{pop_ids.index(pop0)},{pop_ids.index(pop1)}')
        sharex = False
    else:
        sharex = True

    if not cols:
        cols = min(5, num_stats)
    if not rows:
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

    # special cases for color assignment
    if len(models) == 1 and len(means) == 1:
        colors = [palettes.Category10_10[0]] * 2
    else:
        colors = palettes.Category10_10

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
            fig.legend(framealpha=0, loc='lower right', ncols=len(labels))
    if title:
        fig.suptitle(title)

    #fig.legend(
    #   placed_labels, labels, framealpha=0, ncols=2, loc='lower center',
    #    bbox_to_anchor=(0.5, -0.15)
    #)
    if out:
        plt.savefig(out, dpi=244, bbox_inches='tight')
    if show:
        plt.show()

    return 


def plot_parameters(
    pnames, 
    params,
    params_to_plot=None,
    ax_size=1.6,
    out=None,
    show=True
):
    """
    Make a scatter plot of fitted parameters.

    :param pnames: List of parameter names.
    :param params: Array with number of rows equal to the number of fitted 
        models and number of columns equal to the number of fitted parameters.
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
                ax.annotate(name_i, (0.35, 0.5), xycoords='axes fraction',
                            fontsize=14)
                continue 
            x = ax.scatter(params[:, j], params[:, i], marker='o', 
                           c='none', edgecolors='black')
            if i == num_stats - 1:
                ax.tick_params(axis='x', labelrotation=90)
    if out:
        plt.savefig(out, dpi=244, bbox_inches='tight')
    if show:
        plt.show()

    return



def _plot_parameters(
    pnames, 
    params,
    ax_size=2,
    cols=None,
    lls=None,
    out=None,
    show=True
):
    """
    Make a scatter plot of fitted parameters.

    :param pnames: List of parameter names.
    :param params: Array with number of rows equal to the number of fitted 
        models and number of columns equal to the number of fitted parameters.
    """
    mpl.rcParams["xtick.labelsize"] = 7
    mpl.rcParams["ytick.labelsize"] = 7
    mpl.rcParams["legend.fontsize"] = 7
    mpl.rcParams["font.size"] = 7

    assert len(pnames) == params.shape[1]
    if lls is not None:
        assert len(lls) == len(params)

    num_pairs = int(len(pnames) * (len(pnames) - 1) // 2)
    if not cols:
        cols = min(5, num_pairs)
    rows = -(num_pairs // -cols)
    figsize = (cols * ax_size, rows * ax_size)
    fig, axs = plt.subplots(rows, cols, figsize=figsize, layout="constrained")
    if rows > 1:
        axs = axs.flat
    elif cols == 1:
        axs = [axs]
    for ax in axs[num_pairs:]:
        ax.remove()

    k = 0
    for i, name_i in enumerate(pnames):
        for j, name_j in enumerate(pnames):
            if j <= i:
                continue 
            if lls is None:
                c = 'black'
                cmap = None
            else:
                c = lls
                cmap = 'cividis'
            ax = axs[k]
            x = ax.scatter(params[:, i], params[:, j], marker='o', c=c, cmap=cmap)
            ax.set_xlabel(name_i)
            ax.set_ylabel(name_j)
            k += 1

    fig.colorbar(x)

    if out:
        plt.savefig(out, dpi=244, bbox_inches='tight')
    if show:
        plt.show()

    return


def _plot_d_plus_curves(
    stats,
    stats_to_plot=[],
    rows=None,
    cols=None,
    ax_size=2.5,
    dpi=244,
    bins=None,
    rs=None,
    cM=False,
    out_file=None,
    show=True,
):
    """
    DEPRECATED
    """
    statistics = stats.names()
    labels = utils.get_latex_names(stats.pop_ids)
    if len(stats_to_plot) == 0:
        stats_to_plot = stats.names()[0]
    num_stats = len(stats_to_plot)
    if not cols:
        cols = min(5, num_stats)
    if not rows:
        rows = -(num_stats // -cols)
    figsize = (cols * ax_size, rows * ax_size)
    bins = np.asarray(bins)
    mids = (bins[1:] + bins[:-1]) / 2
    x_label = "$r$"
    if cM == True:
        mids = utils.map_function(mids)
        x_label = "cM"

    fig, axs = plt.subplots(rows, cols, figsize=figsize, layout="constrained")
    axs = axs.flat
    for ax in axs[num_stats:]:
        ax.remove()

    for i, stat in enumerate(stats_to_plot):
        ax = axs[i]
        k = statistics[0].index(stat)
        y = [stats[j][k] for j in range(len(mids))]
        ax.plot(mids, y)
        ax.set_xscale("log")
        if i >= (cols * rows - rows):
            ax.set_xlabel(x_label)
        if i % cols == 0:
            ax.set_ylabel("$D^+$")
        label = labels[k]
        ax.set_title(label, y=0.85)

    if out_file:
        plt.savefig(out_file, dpi=dpi)
    if show:
        fig.show()

    return fig


def _plot_empirical_d_plus_curves(
    means,
    varcovs,
    pop_ids=None,
    stats_to_plot=[],
    pops_to_plot=[],
    fill=True,
    rows=None,
    cols=None,
    ax_size=2,
    bins=None,
    rs=None,
    cM=False,
    out=None,
    show=True,
    labels=None
):
    """
    DEPRECATED
    """
    if pop_ids is None:
        raise ValueError('please provide pop_ids')
    statistics = utils.stat_names(pop_ids)
    stat_labels = utils.get_latex_names(pop_ids)
    if len(stats_to_plot) == 0 and len(pops_to_plot) == 0:
        stats_to_plot = statistics[0]
    elif len(stats_to_plot) == 0 and len(pops_to_plot) > 0:
        sorted_pops = [pop for pop in pop_ids if pop in pops_to_plot]
        stats_to_plot = utils.stat_names(sorted_pops)[0]
    num_stats = len(stats_to_plot)
    if not cols:
        cols = min(5, num_stats)
    if not rows:
        rows = -(num_stats // -cols)
    figsize = (cols * ax_size, rows * ax_size)

    bins = np.asarray(bins)
    mids = (bins[1:] + bins[:-1]) / 2
    x_label = "$r$"
    if cM == True:
        mids = utils.map_function(mids)
        x_label = "cM"

    fig, axs = plt.subplots(rows, cols, figsize=figsize, layout="constrained")
    if rows > 1:
        axs = axs.flat
    elif cols == 1:
        axs = [axs]
    for ax in axs[num_stats:]:
        ax.remove()

    if not isinstance(means[0], list):
        means = [means]
        varcovs = [varcovs]

    for k, (_means, _varcovs) in enumerate(zip(means, varcovs)):
        label = labels[k] if labels is not None else None
        color = palettes.Category10_10[k]
        for i, stat in enumerate(stats_to_plot):
            ax = axs[i]
            l = statistics[0].index(stat)
            y_err = np.array(
                [_varcovs[j][l, l] ** 0.5 * 1.96 for j in range(len(mids))]
            )
            y_data = np.array([_means[j][l] for j in range(len(mids))])
            if fill:
                ax.fill_between(
                    mids, y_data - y_err, y_data + y_err, alpha=0.30
                )
                ax.plot(mids, y_data, linestyle="dotted", color=color, label=label)
            else:
                ax.errorbar(
                    mids, 
                    y_data, 
                    yerr=y_err, 
                    capsize=0, 
                    markersize=4,
                    elinewidth=1,
                    fmt="o",
                    mfc="none",
                    mec=color,
                    markeredgewidth=1,
                    ecolor=color,
                    label=label
                )
            if label is not None:
                ax.legend()
            ax.set_xscale("log")
            if i >= (cols * rows - rows):
                ax.set_xlabel(x_label)
            if i % cols == 0:
                ax.set_ylabel("$D^+$")
            stat_label = stat_labels[i]
            ax.set_title(stat_label, y=0.85)

    if out:
        plt.savefig(out, dpi=244)
    if show:
        plt.show()

    return


def plot_gamma_pdf(shape, scale):

    fig, ax = plt.subplots(layout='constrained')
    cutoff = scipy.stats.gamma.ppf(1-1e-5, shape, scale=scale)
    x = np.linspace(0, cutoff, 1000)
    y = scipy.stats.gamma.pdf(x, shape, scale=scale)
    ax.plot(x, y)
    plt.show()

    return 
