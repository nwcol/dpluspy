"""
Houses functions for fitting models to empirical statistics.
"""

from datetime import datetime
import demes
import numpy as np
import moments
from moments.Demes import Inference
import scipy
import sys
import os
import pickle

from . import utils, bootstrapping
from .datastructures import DplusStats


_out_of_bounds = 1e10
_counter = 0


def load_statistics(data_file, graph=None):
    """
    Load bootstrapped statistics stored in a .pkl file, subsetting it to the
    set of populations present in both the file and a Demes graph .yaml file.

    :param data_file: Pathname of a .pkl file holding statistics- minimally
        'pop_ids', 'bins', and corresponding 'means' and 'varcovs'.
    :param graph_file: Optional of a .yaml Demes file- if given, subset to the
        populations common to the graph and data (default None).

    :returns: List of population IDs, bins, means, and varcovs.
    """
    data = pickle.load(open(data_file, "rb"))
    if graph:
        _pop_ids = data["pop_ids"]
        pop_ids = graph_data_overlap(graph, _pop_ids)
        means = bootstrapping.subset_means(data["means"], _pop_ids, pop_ids)
        varcovs = bootstrapping.subset_varcovs(
            data["varcovs"], _pop_ids, pop_ids)
    else:
        pop_ids = data['pop_ids']
        means = data['means']
        varcovs = data['varcovs']
    bins = data["bins"]

    return pop_ids, bins, means, varcovs


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
        to 'simpsons'.
    :param phased: If True, compute phased expectations for cross-population
        statistics (default False).

    :returns: A DPlusStats instance holding computed statistics.
    """
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
        model = DplusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times, 
            rs=midpoints,
            u=u,
            phased=phased
        )

    elif approx == "trapezoid":
        raise ValueError("This method is forbidden as it is too inaccurate")
        y_edges = DplusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times, 
            rs=bins, 
            u=u,
            phased=phased
        )
        y = [(y0 + y1) / 2 for y0, y1 in zip(y_edges[:-2], y_edges[1:-1])]
        y.append(y_edges[-1])
        model = DplusStats(y, pop_ids=sampled_demes)

    elif approx == "simpsons":
        y_edges = DplusStats.from_moments(
            graph, 
            sampled_demes, 
            sample_times=sample_times, 
            rs=bins, 
            u=u,
            phased=phased
        )
        midpoints = (bins[:-1] + bins[1:]) / 2
        y_mids = DplusStats.from_moments(
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
        model = DplusStats(y, pop_ids=sampled_demes)

    else:
        raise ValueError("invalid approximation method!")

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
    fit_ancestral_misid=False
):
    """
    The objective function for model optimization using ``D+``  (and optionally 
    ``H`` or the SFS). 
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

    builder = Inference._update_builder(builder, options, params)
    graph = demes.Graph.fromdict(builder)
    model = compute_bin_stats(
        graph, 
        sampled_demes,
        sample_times=sample_times,
        u=u,
        bins=bins,
        phased=False
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


def _object_func_log(log_p, *args, **kwargs):
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
    misid_guess=None
):
    """
    Optimize a demographic model given empirical ``D+`` statistics and a set of
    model parameters. Wraps functionality from ``moments.Demes.Inference``. 
    Optionally, the allele frequency spectrum (AFS) or the pairwise diversity
    statistic ``H`` may be jointly used in the optimization.

    :param graph_file: Pathname of file holding ``demes``-format model to fit.
    :param param_file: Pathname of parameter file, specified as described at 
        https://momentsld.github.io/moments/extensions/demes.html#the-options-file
    :param means: List of ``D+`` statistics obtained using `bootstrapping`. 
        ``H`` statistics are the last element of this list.
    :param varcovs: List of variance-covariance matrices from `bootstrapping`.
    :param pop_ids: List of population IDs corresponding to `means`.
    :param bins: Array of recombination bin edges in units of ``r``.
    :param u: Mutation rate parameter. If fitting the mutation rate, provides 
        the initial guess for this parameter (here defaults to 1e-8).
    :param method: Optimization algorithm to use.
    :param max_iter: Maximum number of optimization iterations.
    :param log: If True, optimize over the log of parameters (default False)
    :param verbose: Print convergence messsages every `verbose` function calls.
    :param overwrite: If True, overwrites existing files with output 
        (default False).
    :param output: Pathname at which to write fitted graph file.
    :param use_H: If True, include ``H`` in optimization (default False).
    :param use_afs: If True, include the allele frequency spectrum `afs` in the 
        fit (default False).
    :param afs: AFS (SFS) data to use in fitting.
    :param L: Required when fitting the AFS; gives corresponding sequence 
        length.
    :param perturb: Perturb initial parameters by up to `perturb`-fold (default
        0 does not perturb parameters).
    :param fit_mutation_rate: If True, fit the mutation rate as an additional
        parameter (default False).
    :param u_bounds: When fitting the mutation rate, provides upper and lower
        bounds for that parameter (defaults to (5e-9, 2e-8)).
    :param fit_ancestral_misid: When fitting jointly with an unfolded AFS and
        True, fit the probability that the ancestral state is misspecified
        (default False).
    :param misid_guess: Initial guess for the misid probability (defaults to 
        0.02).

    :returns: List of parameter names, optimized values, and log-likelihood.
    """
    builder = Inference._get_demes_dict(graph_file)
    options = Inference._get_params_dict(param_file)
    params_bounds = Inference._set_up_params_and_bounds(options, builder)
    param_names, params_0, lower_bounds, upper_bounds = params_bounds
    constraints = Inference._set_up_constraints(options, param_names)

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
        params_0 = Inference._perturb_params_constrained(
            params_0, 
            perturb, 
            lower_bound=lower_bounds, 
            upper_bound=upper_bounds,
            cons=constraints
        )
    if log:
        objective = _object_func_log
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
    
    warn = None
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
        fit_ancestral_misid
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
        fit_params, fopt, num_iter, func_calls, flag = ret[:5]

    elif method == 'powell':
        ret = scipy.optimize.fmin_powell(
            objective,
            params_0,
            args=args,
            maxiter=max_iter,
            full_output=True,
        )
        fit_params, fopt, _, num_iter, func_calls, flag = ret[:6]

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
        fit_params, fopt, _, __, func_calls, grad_calls, flag = ret[:7]
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
        warn = output_dict["task"]

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
        builder = Inference._update_builder(builder, options, fit_params)
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


def print_status(n_calls, ll, params):
    """
    Print the number of function calls, the log-likelihood, and the current 
    parameter values.
    """
    t = utils._current_time()
    _n = f'{n_calls:<4}'
    if isinstance(ll, float):
        _ll = f'{np.round(ll, 2):>10}'
    else:
        _ll = f'{ll:>10}'
    fmt_p = []
    for x in params:
        if isinstance(x, str):
            fmt_p.append(f'{x:>10}')
        else:
            if x > 1:
                fmt_p.append(f'{np.round(x, 1):>10}')
            else:
                sci = np.format_float_scientific(x, 2, trim='k')
                fmt_p.append(f'{sci:>10}')
    _p = ''.join(fmt_p)
    print(t, _n, _ll, '[', _p, ']')

    return


def format_params(params):
    """
    returns strings
    
    """
    formatted = []
    for param in params:
        if param >= 1:
            formatted.append(str(np.round(param, 1)))
        elif param >= 1e-3:
            formatted.append(np.format_float_positional(param, precision=3))
        else:
            formatted.append(np.format_float_scientific(param, precision=2))

    return formatted


## computing log-likelihoods


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
    for i in range(n_bins):
        if (
            i in _inv_varcov_cache  
            and np.all(_inv_varcov_cache[i]["varcov"] == varcovs[i])
        ):
            inv_varcov = _inv_varcov_cache[i]["inv_varcov"]
        else:
            inv_varcov = np.linalg.inv(varcovs[i])
            add_to_cache = {"varcov": varcovs[i], "inv_varcov": inv_varcov}
            _inv_varcov_cache[i] = add_to_cache
        bin_ll[i] = _ll(xs[i], mus[i], inv_varcov)

    return bin_ll


def _ll(x, mu, inv_varcov):
    """
    Compute the log of the multivariate gaussian function with means `mu`, 
    pre-inverted covariance matrix `inv_cov` at `x`.

    :param x: Array at which to evaluate the function.
    :type x: np.ndarray, shape (n,)
    :param mu: Array specifying mean parameters for the distribution.
    :type mu: np.ndarray, shape (n,)
    :inv_varcov: Pre-inverted covariance matrix parameterizing the distribution
    :type inv_varcov: np.ndarray, shape (n, n)

    :returns: Log of the multivariate gaussian law.
    """
    return -1.0 / 2.0 * np.matmul(np.matmul(x - mu, inv_varcov), x - mu)


def _log_multivariate_normal_pdf(x, mu, varcov):
    """
    Evaluate the log of the complete multivariate normal probability law.
    """
    f = _multivariate_normal_pdf(x, mu, varcov)
    return np.log(f)


def _multivariate_normal_pdf(x, mu, varcov):
    """
    Evaluate the full multivariate normal PDF with means `mu`, variance-
    covariances `varcov` and point `x`.
    """
    k = len(x)
    inv_varcov = np.linalg.inv(varcov)
    f = (
        np.exp(-1.0 / 2.0 * np.matmul(np.matmul(x - mu, inv_varcov), x - mu)) 
        / np.sqrt(np.linalg.det(varcov) * (2.0 * np.pi) ** k) 
    )
    return f


## Computing uncertainties; estimating standard errors


def compute_uncerts(
    graph_file,
    param_file,
    means,
    varcovs,
    pop_ids=None,
    bins=None,
    u=None,
    fitted_u=None,
    bootstrap_reps=None,
    delta=0.01,
    method="godambe"
):
    """
    Compute parameter estimates using either the Fisher information matrix 
    ('fisher' method) or the Godambe information matrix ('godambe'). 

    This function is adopted from the Godambe uncertainty estimators already 
    implemented in `moments`.

    :param graph_file: Pathname of a `demes`-format .yaml file specifying a
        fitted demographic model.
    :param param_file: Pathname of an options file.
    :param means: List of arrays holding binned mean statistics.
    :param varcovs: List of variance-covariance matrices.
    :param pop_ids: List of population IDs.
    :param bins: Array of recombination distance bin edges.
    :param u: Fixed mutation rate parameter. Mutually exclusive with `fitted_u`.
    :param fitted_u: Fitted mutation rate parameter. This value is appended
        to the fitted parameters and uncerts are computed for it as well.
    :param bootstrap_reps: List of bootstrap replicate means. Required when
        `method` is 'godambe' and otherwise not used.
    :param delta: Step size for evaluating the gradient, etc with finite
        differences.
    :param method: Method to use for computing standard deviations. Can be 
        either 'fisher'- which does not require bootstrap replicates but 
        understimates variance because genetic linkage violates assumptions-
        or 'godambe', which requires bootstrap replicates.

    :returns: A list of parameter names, of input parameter values, and 
        estimated standard deviations of parameter estimates.
    """
    if method not in ("godambe", "fisher"):
        raise ValueError("invalid method")

    builder = Inference._get_demes_dict(graph_file)
    options = Inference._get_params_dict(param_file)
    params_bounds = Inference._set_up_params_and_bounds(options, builder)
    param_names, params = params_bounds[:2]

    if fitted_u is not None:
        if u is not None:
            raise ValueError('You cannot specify both `u` and `fitted_u`')
        param_names.append('u')
        params = np.append(params, fitted_u)
        fitted_mutation_rate = True
    else:
        fitted_mutation_rate = False

    deme_names = [d["name"] for d in builder["demes"]]
    sampled_demes = [] 
    sample_times = []
    for pop in pop_ids: 
        assert pop in deme_names
        idx = deme_names.index(pop)
        sample_times.append(builder["demes"][idx]["epochs"][-1]["end_time"])
        sampled_demes.append(pop)

    model_args = (
        builder, 
        options, 
        sampled_demes, 
        sample_times, 
        bins, 
        u,
        fitted_mutation_rate
    )

    def model_func(params, args=()):
        """
        Compute expected ``D+`` given `params`.
        """
        (
            builder, 
            options, 
            sampled_demes, 
            sample_times, 
            bins, 
            u, 
            fitted_mutation_rate
        ) = args
        if fitted_mutation_rate:
            u = params[-1]
        builder = Inference._update_builder(builder, options, params)
        graph = demes.Graph.fromdict(builder)
        model = compute_bin_stats(
            graph, 
            sampled_demes,
            sample_times=sample_times,
            u=u,
            bins=bins,
            phased=False
        )
        return model

    if method == "fisher":
        H = _compute_godambe_matrix(
            params,
            model_func,
            model_args,
            means,
            varcovs,
            None,
            delta=delta,
            get_hessian=True
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(H)))

    elif method == "godambe":
        if bootstrap_reps is None:
            raise ValueError('we need bootstrap_reps to use `godambe` method!')
        G, _, __ = _compute_godambe_matrix(
            params,
            model_func,
            model_args,
            means,
            varcovs,
            bootstrap_reps,
            delta=delta,
            get_hessian=False
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(G)))
    else:
        return

    return param_names, params, uncerts


_model_cache = dict()


def _compute_godambe_matrix(
    params_0,
    model_func,
    model_args,
    means,
    varcovs,
    bootstrap_reps,
    delta=0.01,
    get_hessian=False,
    verbose=False
):
    """
    Compute the Fisher (FIM) or Godambe (GIM) information matrix. These objects
    are used to compute the uncertainties of parameters inferred with (here, 
    composite) maximum likelihood.
    """
    def obj_func(params, means, varcovs, model_args):
        
        key = tuple(params)
        if key in _model_cache:
            model = _model_cache[key]
        else:
            model = model_func(params, model_args)
            _model_cache[key] = model

        return composite_ll(model, means, varcovs)

    H = - _compute_hessian(
        params_0, 
        obj_func, 
        model_args,
        means,
        varcovs,
        delta=delta
    )
    if verbose:
        print(_current_time(), "computed Hessian")

    if get_hessian:
        return H

    J = np.zeros((len(params_0), len(params_0)))
    for i, bootmeans in enumerate(bootstrap_reps):
        cU = _compute_gradient(
            params_0, 
            obj_func, 
            model_args,
            bootmeans,
            varcovs,
            delta=delta
        )
        if verbose:
            print(_current_time(), f"computed gradient for bootstrap set {i}")
        cJ = np.matmul(cU, cU.T)
        J += cJ
    J = J / len(bootstrap_reps)
    J_inv = np.linalg.inv(J)
    G = np.matmul(np.matmul(H,  J_inv), H)

    return G, H, J


def _compute_hessian(p0, obj_func, model_args, means, varcovs, delta=0.01):
    """
    Compute the approximate Hessian matrix of the log-likelihood function. Uses 
    empirical means and varcovs obtained by bootstrapping. 
    """
    f0 = obj_func(p0, means, varcovs, model_args)
    hs = delta * p0
    H = np.zeros((len(p0), len(p0)), dtype=np.float64)
    for i in range(len(p0)):
        for j in range(i, len(p0)):
            p = np.array(p0, copy=True, dtype=np.float64)
            if i == j:
                p[i] = p0[i] + hs[i]
                fp = obj_func(p, means, varcovs, model_args)
                p[i] = p0[i] - hs[i]
                fm = obj_func(p, means, varcovs, model_args)
                element = (fp - 2 * f0 + fm) / hs[i] ** 2
            else:
                p[i] = p0[i] + hs[i]
                p[j] = p0[j] + hs[j]
                fpp = obj_func(p, means, varcovs, model_args)
                p[i] = p0[i] + hs[i]
                p[j] = p0[j] - hs[j]
                fpm = obj_func(p, means, varcovs, model_args)
                p[i] = p0[i] - hs[i]
                p[j] = p0[j] + hs[j]
                fmp = obj_func(p, means, varcovs, model_args)
                p[i] = p0[i] - hs[i]
                p[j] = p0[j] - hs[j]
                fmm = obj_func(p, means, varcovs, model_args)
                element = (fpp - fpm - fmp + fmm) / (4 * hs[i] * hs[j])
            H[i, j] = element
            H[j, i] = element

    return H


def _compute_gradient(p0, obj_func, model_args, means, varcovs, delta=0.01):
    """
    Compute the gradient of the log likelihood function. 
    """
    hs = delta * p0
    gradient = np.zeros((len(p0), 1))
    for i in range(len(p0)):
        p = np.array(p0, copy=True, dtype=float)
        p[i] = p0[i] + hs[i]
        fp = obj_func(p, means, varcovs, model_args)
        p[i] = p0[i] - hs[i]
        fm = obj_func(p, means, varcovs, model_args)
        gradient[i] = (fp - fm) / (2 * hs[i])

    return gradient


## Utilities and printing functions


def graph_data_overlap(graph, pop_ids):
    """
    Find the populations which occur mutually in a Demes graph and a list of 
    population names.

    :param graph: Demes graph or the path and file name leading to a .yaml file
        specifying a demes graph. 
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


def _load_parameters(graph_file, param_file):
    """
    Load a list of parameter names and a vector of their values from a graph 
    file.
    """
    builder = Inference._get_demes_dict(graph_file)
    options = Inference._get_params_dict(param_file)
    pnames, params, *_  = Inference._set_up_params_and_bounds(options, builder)
    
    return pnames, params

