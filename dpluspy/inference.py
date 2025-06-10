"""
Houses functions for fitting models to empirical statistics.
"""

from datetime import datetime
import demes
import numpy as np
import moments
import scipy
import os
import pickle

from . import bootstrapping
from .datastructures import DPlusStats


_out_of_bounds = 1e10
_counter = 0


def load_stats(data_file, graph=None, to_pops=None):
    """
    Load bootstrapped statistics stored in a .pkl file, subsetting it to the
    set of populations present in both the file and a Demes graph .yaml file.

    :param data_file: Pathname of a .pkl file holding statistics- minimally
        'pop_ids', 'bins', and corresponding 'means' and 'varcovs'.
    :param graph_file: Optional pathname of a .yaml Demes file- if given, 
        subsets to the populations common to graph and data (default None).
    :param to_pops: Optional list of populations to subset to.

    :returns: List of population IDs, bins, means, and varcovs.
    """
    with open(data_file, "rb") as fin:
        data = pickle.load(fin)
    _pop_ids = data["pop_ids"]
    if graph is not None:
        to_pops = graph_data_overlap(graph, _pop_ids)
    if to_pops is not None:
        means = bootstrapping.subset_means(data["means"], _pop_ids, to_pops)
        varcovs = bootstrapping.subset_varcovs(
            data["varcovs"], _pop_ids, to_pops)
        pop_ids = to_pops
    else:
        pop_ids = data['pop_ids']
        means = data['means']
        varcovs = data['varcovs']
    bins = data["bins"]
    return pop_ids, bins, means, varcovs


def load_bootstrap_reps(filename, graph=None, to_pops=None, num_reps=None):
    """
    Load varcovs, means and replicate means from a pickled dictionary with
    keys "pop_ids", "varcovs", "means", "bins" and "replicates". "replicates"
    should be a list of bootstrap replicate means.

    :param str graph: Pathname of a demes-format YAML file.
    :param str filename: Pathname of .pkl file.
    :parma list to_pops: Optional list of populations to subset to (default 
        None)
    :param int num_reps: Optional number of bootstrap replicates to load 
        (if None, returns all replicates). 

    :rtype: dict
    """
    with open(filename, "rb") as fin:
        archive = pickle.load(fin)
    pop_ids = archive["pop_ids"]
    bins = archive["bins"]
    means = archive["means"]
    varcovs = archive["varcovs"]
    replicates = archive["replicates"]
    if num_reps is not None:
        if num_reps > len(replicates):
            raise ValueError("`num_reps` exceeds number of replicates")
    if graph is not None:
        to_pops = graph_data_overlap(graph, pop_ids)
    if to_pops is not None:
        means = bootstrapping.subset_means(means, pop_ids, to_pops)
        varcovs = bootstrapping.subset_varcovs(varcovs, pop_ids, to_pops)
        replicates = [bootstrapping.subset_means(rep, pop_ids, to_pops) 
                      for rep in replicates]
        pop_ids = to_pops
    ret = {
        "bins": bins,
        "pop_ids": pop_ids,
        "means": means,
        "varcovs": varcovs,
        "replicates": replicates
    }
    return ret


def compute_bin_stats(
    graph,
    sampled_demes, 
    sample_times=None, 
    u=None,
    bins=None,
    approx="simpsons",
    phased=False
):
    """
    From a Demes graph, compute expected ``D+`` in bins using `moments.LD` and 
    a given approximation method. This is effectively a wrapped for the  
    `moments.Demes.LD()` function. `bins` must be provided.

    :param param graph: `demes` graph or pathname of a .yaml file specifying a 
        demographuc model in the `demes` format.
    :param sampled_demes: List of demes to compute statistics for.
    :param sample_times: Optional list of sample times for demes. Default to 
        the specified end times.
    :param u: Mutation rate parameter (defaults to 1).
    :param bins: Recombination distance bin edges, in units of ``r``.
    :param approx: Method for approximating quantities in each bin; defaults
        to 'simpsons' if None.
    :param phased: If True, compute phased expectations for cross-population
        statistics (default False).

    :returns: A DPlusStats instance holding computed statistics.
    """
    if approx is None:
        approx = "simpsons"

    if approx not in ("midpoint", "trapezoid", "simpsons"):
        raise ValueError("Unrecognized approximation method")
    
    if bins is None:
        raise ValueError('You must provide bins')

    if isinstance(graph, str):
        graph = demes.load(graph)

    if u is None:
        u = 1

    if approx == "midpoint":
        midpoints = (bins[:-1] + bins[1:]) / 2
        model = DPlusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times, 
            rs=midpoints,
            u=u,
            phased=phased
        )

    elif approx == "trapezoid":
        raise ValueError("This method is forbidden as it is too inaccurate")
        y_edges = DPlusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times, 
            rs=bins, 
            u=u,
            phased=phased
        )
        y = [(y0 + y1) / 2 for y0, y1 in zip(y_edges[:-2], y_edges[1:-1])]
        y.append(y_edges[-1])
        model = DPlusStats(y, pop_ids=sampled_demes)

    elif approx == "simpsons":
        y_edges = DPlusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times, 
            rs=bins, 
            u=u,
            phased=phased
        )
        midpoints = (bins[:-1] + bins[1:]) / 2
        y_mids = DPlusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times,
            rs=midpoints, 
            u=u,
            phased=phased
        )        
        y = [
            (y_edges[i] + 4 * y_mids[i] + y_edges[i + 1]) / 6 
            for i in range(len(midpoints))
        ]
        y.append(y_edges[-1])
        model = DPlusStats(y, pop_ids=sampled_demes)

    else:
        raise ValueError("Unrecognized approximation method")
    return model


## Optimization functions


def _object_func(
    params,
    builder,
    options,
    means,
    varcovs,
    sampled_demes=None,
    sample_times=None,
    u=None,
    bins=None,
    lower_bounds=None,
    upper_bounds=None,
    constraints=None,
    verbose=None,
    use_H=False,
    use_afs=False,
    afs=None,
    L=None,
    fit_mutation_rate=False,
    fit_ancestral_misid=False,
    approx_method=None
):
    """
    The objective function for model optimization using D+ (and optionally H
    or the SFS). 
    """
    if lower_bounds is not None and np.any(params < lower_bounds):
        return _out_of_bounds
    elif upper_bounds is not None and np.any(params > upper_bounds):
        return _out_of_bounds
    elif constraints is not None and np.any(constraints(params) <= 0):
        return _out_of_bounds

    global _counter
    _counter += 1    

    if fit_mutation_rate:
        if fit_ancestral_misid:
            u = params[-2]
        else:
            u = params[-1]

    builder = moments.Demes.Inference._update_builder(builder, options, params)
    graph = demes.Graph.fromdict(builder)
    model = compute_bin_stats(
        graph, 
        sampled_demes,
        sample_times=sample_times,
        u=u,
        bins=bins,
        phased=False,
        approx=approx_method
    )
    ll = composite_ll(model, means, varcovs, use_H=use_H)

    if use_afs:
        sample_sizes = afs.sample_sizes
        model_afs = moments.Demes.SFS(
            graph, 
            sampled_demes, 
            sample_sizes, 
            sample_times=sample_times, 
            u=u * L
        )
        if fit_ancestral_misid:
            p_misid = params[-1]
            model_afs = moments.Misc.flip_ancestral_misid(model_afs, p_misid)
        ll += moments.Inference.ll(model_afs, afs)
    
    if verbose > 0 and _counter % verbose == 0:
        pstr = ''.join([f'{float(p):>10.3}' for p in params])
        print(f'{_counter:<5}{np.round(ll, 2):>10} [{pstr}]')
    return -ll


def _log_object_func(log_p, *args, **kwargs):
    """
    Objective function for optimizing over the log of parameters.
    """
    p = np.exp(log_p - 1)
    return _object_func(p, *args, **kwargs)


def optimize(
    graph_file,
    param_file,
    means,
    varcovs,
    pop_ids=None,
    bins=None,
    u=None,
    method="fmin",
    max_iter=1000,
    log=False,
    verbose=1,
    overwrite=False,
    output=None,
    use_H=False,
    use_afs=False,
    afs=None,
    L=None,
    perturb=False,
    fit_mutation_rate=False,
    u_bounds=None,
    fit_ancestral_misid=False,
    misid_guess=None,
    approx_method=None
):
    """
    Fit a demographic model to observed D+ statistics using composite maximum 
    likelihood. Demographic models are expressed in Demes format and parameters
    are specified as in moments.Demes: 
    https://momentsld.github.io/moments/extensions/demes.html#the-options-file

    Largely replicates or wraps functionality from moments.Demes, but for the
    D+ statistic specifically. H statistics or the SFS/AFS may optionally also
    be included in the fit.

    :param str graph_file: Pathname of YAML file holding Demes-format model.
    :param str param_file: Pathname of YAML parameter file.
    :param list means: Bin-wise list of mean empirical D+ statistics. The last
        entry should be an array of H statistics.
    :param list varcovs: List of covariance matrices obtained via bootstrap.
    :param list pop_ids: Required list of population IDs.
    :param arr bins: Array of recombination bin edges in units of r.
    :param float u: Mutation rate parameter. If fitting the mutation rate, gives 
        the initial guess for this parameter (defaults to 1e-8).
    :param str method: Optimization algorithm to use (default "fmin").
    :param int max_iter: Maximum number of optimization iterations.
    :param bool log: If True, optimize over the log of params (default False)
    :param int verbose: Print convergence messsages every `verbose` function 
        calls.
    :param bool overwrite: If True, overwrites existing files with output 
        (default False).
    :param str output: Pathname to write fitted graph file.
    :param bool use_H: If True, fit H statistics as well as D+ (default False).
    :param vool use_afs: If True, fit to the allele frequency spectrum `afs` 
        (default False). Requires that `afs` and `L` are given.
    :param moments.Spectrum afs: AFS (SFS) data to use in fitting.
    :param int L: Effective sequence length, required when fitting the AFS.
    :param float perturb: Perturb initial parameters by up to `perturb`-fold 
        (default 0 does not perturb parameters).
    :param bool fit_mutation_rate: If True, fits the mutation rate as a free
        parameter (default False).
    :param tuple u_bounds: When fitting the mutation rate, provides upper and 
        lower bounds for that parameter (defaults to (5e-9, 2e-8)).
    :param bool fit_ancestral_misid: When fitting jointly with an unfolded AFS 
        and True, fits the probability that the ancestral state is misspecified
        (default False).
    :param float misid_guess: Initial guess for the misid probability (defaults 
        to 0.02).
    :param str approx_method: Optional method to use for approximating E[D+] 
        within bins (defaults to "simpsons"). The other option is "midpoint",
        which is about two times faster but slightly less precise.

    :returns tuple: List of parameter names, list of fitted parameter values, 
        and log-likelihood.
    """
    builder = moments.Demes.Inference._get_demes_dict(graph_file)
    options = moments.Demes.Inference._get_params_dict(param_file)
    params_bounds = moments.Demes.Inference._set_up_params_and_bounds(
        options, builder)
    param_names, params_0, lower_bounds, upper_bounds = params_bounds
    constraints = moments.Demes.Inference._set_up_constraints(
        options, param_names)

    if u is None and not fit_mutation_rate:
        raise ValueError("You must provide `u`")
    if pop_ids is None:
        raise ValueError("You must provide `pop_ids`")
    
    if use_afs:
        if afs is None:
            raise ValueError('You must provide `afs` to use `fit_afs`')
        if L is None:
            raise ValueError('You must provide `L` to use `fit_afs`')
    
    if fit_mutation_rate:
        if u is None:
            u = 1e-8
        param_names.append('u')
        params_0 = np.append(params_0, u)
        if u_bounds is None:
            u_bounds = (5e-9, 2e-8)
        lower_bounds = np.append(lower_bounds, u_bounds[0])
        upper_bounds = np.append(upper_bounds, u_bounds[1])

    if fit_ancestral_misid:
        if not use_afs:
            raise ValueError('You must fit the AFS to `fit_ancestral_misid`')
        if afs.folded:
            raise ValueError('The AFS is folded: cannot `fit_ancestral_misid`')
        param_names.append('p_misid')
        if misid_guess is None:
            misid_guess = 0.02
        param_names.append('p_misid')
        params_0 = np.append(params_0, misid_guess)
        lower_bounds = np.append(lower_bounds, 0)
        upper_bounds = np.append(upper_bounds, 1)
    
    if perturb > 0: 
        params_0 = moments.Demes.Inference._perturb_params_constrained(
            params_0, 
            perturb, 
            lower_bound=lower_bounds, 
            upper_bound=upper_bounds,
            cons=constraints
        )
    if log:
        objective = _log_object_func
        params_0 = np.log(params_0) + 1
    else:
        objective = _object_func
    
    print(_current_time(), f"Fitting D+ to data for {pop_ids}")
    namestr = ''.join([f'{n:>10}' for n in param_names])
    pstr = ''.join([f'{float(p):>10.3}' for p in params_0])
    print(f'{"Call":<5}{"LL":>10} [{namestr}]')
    print(f'{"init":<5}{"-":>10} [{pstr}]')

    deme_names = [d["name"] for d in builder["demes"]]
    sampled_demes = [] 
    sample_times = []
    for pop in pop_ids: 
        assert pop in deme_names
        idx = deme_names.index(pop)
        sample_times.append(builder["demes"][idx]["epochs"][-1]["end_time"])
        sampled_demes.append(pop)
    
    args = (
        builder,
        options,
        means,
        varcovs,
        sampled_demes,
        sample_times,
        u,
        bins,
        lower_bounds,
        upper_bounds,
        constraints,
        verbose,
        use_H,
        use_afs,
        afs,
        L,
        fit_mutation_rate,
        fit_ancestral_misid,
        approx_method
    )
    
    methods = ['fmin', 'powell', 'bfgs', 'lbfgsb']
    if method not in methods:
        raise ValueError(f"{method} is not a valid method")
    
    if method == 'fmin':
        ret = scipy.optimize.fmin(
            objective,
            params_0,
            args=args,
            maxiter=max_iter,
            full_output=True
        )
        fit_params, fopt, num_iter, _, flag = ret[:5]

    elif method == 'powell':
        ret = scipy.optimize.fmin_powell(
            objective,
            params_0,
            args=args,
            maxiter=max_iter,
            full_output=True,
        )
        fit_params, fopt, _, num_iter, __, flag = ret[:6]

    elif method == 'bfgs':
        if log:
            epsilon = 1e-3
        else:
            epsilon = None
        ret = scipy.optimize.fmin_bfgs(
            objective,
            params_0,
            args=args,
            maxiter=max_iter,
            epsilon=epsilon,
            disp=False,
            full_output=True
        )
        fit_params, fopt, _, __, ___, grad_calls, flag = ret[:7]
        num_iter = grad_calls

    elif method == 'lbfgsb':
        if log:
            bounds = list(
                zip(np.log(lower_bounds) + 1, np.log(upper_bounds) + 1)
            )
            epsilon = 1e-5
            pgtol = 1e-5
        else:
            bounds = list(zip(lower_bounds, upper_bounds))
            epsilon = 1e-8
            pgtol = 1e-5
        ret = scipy.optimize.fmin_l_bfgs_b(
            objective,
            params_0,
            args=args,
            maxiter=max_iter,
            bounds=bounds,
            epsilon=epsilon,
            pgtol=pgtol,
            approx_grad=True
        )
        fit_params, fopt, output_dict = ret
        num_iter = output_dict['nit']
        flag = output_dict['warnflag']

    else:
        return

    if log: 
        fit_params = np.exp(fit_params - 1)

    ll = -fopt

    print(f"Log-likelihood:\t{np.round(ll, 2)}")
    print("Fitted parameters:")
    for name, value in zip(param_names, fit_params):
        print(f"{name}\t{value:.3}")

    global _counter

    if output is not None:
        builder = moments.Demes.Inference._update_builder(
            builder, options, fit_params)
        graph = demes.Graph.fromdict(builder)
        # Record some information about the fit in the 'metadata' field
        info = {
            'll': ll,
            'num_iter': num_iter,
            'max_iter': max_iter,
            'flag': flag
        }
        if fit_mutation_rate:
            if fit_ancestral_misid:
                fitted_u = fit_params[-2]
            else:
                fitted_u = fit_params[-1]
            info['fitted_u'] = fitted_u
        else:
            info['u'] = u
        if fit_ancestral_misid:
            info['fitted_misid'] = fit_params[-1]
        graph.metadata['opt_info'] = info

        if overwrite is False and os.path.isfile(output):
            print(f"{output} already exists: printing model")
            print(str(graph))
        else:
            demes.dump(graph, output)
    _counter = 0
    return param_names, fit_params, ll


_inv_varcov_cache = dict()


def composite_ll(model, means, varcovs, use_H=False):
    """
    Compute the sum of log-likelihoods across ``D+`` bins.
    """
    if use_H:
        ll = ll_per_bin(model, means, varcovs).sum()
    else:
        ll = ll_per_bin(model[:-1], means[:-1], varcovs[:-1]).sum()
    return ll


def ll_per_bin(xs, mus, varcovs):
    """
    Compute LL in each bin and return an array of bin LLs.
    """
    n_bins = len(xs)
    if len(mus) != n_bins or len(varcovs) != n_bins:
        raise ValueError("Data, model and varcovs must have the same length")
    bin_ll = np.zeros(n_bins, dtype=np.float64)
    for ii in range(n_bins):
        if (
            ii in _inv_varcov_cache  
            and np.all(_inv_varcov_cache[ii]["varcov"] == varcovs[ii])
        ):
            inv_varcov = _inv_varcov_cache[ii]["inv_varcov"]
        else:
            inv_varcov = np.linalg.inv(varcovs[ii])
            add_to_cache = {"varcov": varcovs[ii], "inv_varcov": inv_varcov}
            _inv_varcov_cache[ii] = add_to_cache
        bin_ll[ii] = _ll(xs[ii], mus[ii], inv_varcov)
    return bin_ll


def _ll(x, mu, inv_varcov):
    """
    Compute the log of the multivariate gaussian function with means `mu`, 
    pre-inverted covariance matrix `inv_cov` at `x`. Drops the coefficient.

    :param np.ndarray x: Empirical means
    :param np.ndarray mu: Model expectations
    :param np.ndarray inv_varcov: Pre-inverted covariance matrix obtained from
        a bootstrap over genomic regions

    :returns float: Log of the multivariate gaussian law.
    """
    return -0.5 * np.matmul(np.matmul((x - mu).T, inv_varcov), x - mu)


def exact_ll_per_bin(xs, mus, varcovs):
    """
    Compute the log-likelihood in each bin without dropping coefficients.
    """ 
    n_bins = len(xs)
    bin_ll = np.zeros(n_bins, np.float64)
    for ii in range(n_bins):
        bin_ll[ii] = scipy.stats.multivariate_normal(
            mean=mus[ii], cov=varcovs[ii]).logpdf(xs[ii])
    return bin_ll


def graph_data_overlap(graph, pop_ids):
    """
    Find the intersection of the sets of populations in `pop_ids` and deme
    names in `graph`.

    :returns list: Names of populations which occur in both inputs.
    """
    if isinstance(graph, str):
        graph = demes.load(graph)
    deme_names = [d.name for d in graph.demes]
    overlaps = [pop for pop in pop_ids if pop in deme_names]
    return overlaps


def _current_time():
    """
    Get a string representing the date and time.
    """
    return "[" + datetime.strftime(datetime.now(), "%d-%m-%y %H:%M:%S") + "]"


def _load_params(graph_file, param_file):
    """
    Load a list of parameter names and a vector of their values from a graph 
    file.
    """
    builder = moments.Demes.Inference._get_demes_dict(graph_file)
    options = moments.Demes.Inference._get_params_dict(param_file)
    pnames, params, *_  = moments.Demes.Inference._set_up_params_and_bounds(
        options, builder)
    return pnames, params
