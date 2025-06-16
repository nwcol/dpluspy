"""
Functions for estimating confidence intervals and statistical testing.
"""

from datetime import datetime
import demes
import numpy as np
import moments
import scipy
import sys
import os
import pickle

from . import inference
from .datastructures import DPlusStats


def _model_func(params, args=()):
    """
    Compute expected D+ given a vector of `params` and other arguments.

    :param array params: Array of parameter values.
    :param tuple args: Other arguments required to evaluate model expectations;
        see unpacking of `args` below. Can be created with `set_up_model_args`.
        These arguments are static and are not altered as we compute the score,
        Hessian matrix etc. 
    
    :returns list: Bin-wise list of expected D+ statistics.
    """
    (
        builder, 
        options, 
        sampled_demes, 
        sample_times, 
        bins, 
        u, 
        fitted_mutation_rate,
        approx_method
    ) = args
    if fitted_mutation_rate:
        u = params[-1]
    builder = moments.Demes.Inference._update_builder(
        builder, options, params)
    graph = demes.Graph.fromdict(builder)
    model = inference.compute_bin_stats(
        graph, 
        sampled_demes,
        sample_times=sample_times,
        u=u,
        bins=bins,
        phased=False,
        approx=approx_method
    )
    return model


def _set_up_model_args(
    graph_file,
    param_file,
    pop_ids=None,
    bins=None,
    u=None,
    fitted_u=None,
    approx_method=None,
):
    """
    Load best-fit parameters from a Demes file and set up a tuple of arguments 
    for `_model_func`.

    :param str graph_file: Pathname of Demes-specification YAML file encoding
        max-likelihood best-fit demographic model.
    :param str param_file: Pathname of YAML file encoding parameter spec.,
        in the form used by moments.Demes.
    :param list pop_ids: List of population IDs (required).
    :param array bins: Array of recombination bin edges (required).
    :param float u: Fixed mutation rate parameter. Either `u` or `fitted_u` must 
        be given.
    :param float fitted_u: Best-fit fitted mutation rate parameter. 
    :param str approx_method: Method to use when approximating average D+ in 
        bins ("midpoint" or "simpsons". Defaults to "simpsons").
        
    :returns tuple: Array of parameter values, list of parameter names, and 
        tuple of static arguments.
    """  
    builder = moments.Demes.Inference._get_demes_dict(graph_file)
    options = moments.Demes.Inference._get_params_dict(param_file)
    params_bounds = moments.Demes.Inference._set_up_params_and_bounds(
        options, builder)
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
        fitted_mutation_rate,
        approx_method
    )
    return params, param_names, model_args


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
    method="godambe",
    approx_method=None,
    model_func=_model_func,
    verbose=True
):
    """
    Compute parameter estimates using either the Fisher information matrix (FIM,
    "fisher" method) or the Godambe information matrix (GIM, "godambe" method). 

    This function is adapted from the Godambe uncertainty estimators already 
    implemented in `moments` and the methods of Coffman et al. 2015.

    :param str graph_file: Pathname of a `demes`-format .yaml file specifying a
        fitted demographic model.
    :param str param_file: Pathname of an options file.
    :param list means: List of arrays holding binned mean statistics.
    :param list varcovs: List of variance-covariance matrices.
    :param list pop_ids: List of population IDs.
    :param array bins: Array of recombination distance bin edges.
    :param float u: Fixed mutation rate parameter. Mutually exclusive with 
        `fitted_u`.
    :param float fitted_u: Fitted mutation rate parameter. This value is
        appended to fitted parameters and uncerts are computed for it as well.
    :param list bootstrap_reps: List of bootstrap replicate means. Required when
        `method` is "godambe", otherwise not used.
    :param float delta: Optional step size for evaluating derivatives with 
        finite differences (default 0.01).
    :param str method: Method to use for computing standard deviations. Can be 
        either "fisher"- which does not require bootstrap replicates but 
        understimates variance because genetic linkage violates its assumptions-
        or "godambe", which requires `bootstrap_reps`.
    :param str approx_method: Optional method to use for approximating E[D+] 
        within bins (defaults to "simpsons"). The other option is "midpoint",
        which is about two times faster but slightly less precise.
    :param function model_func: Function to use when evaluating model 
        expectations (default `_model_func`). 
    :param verbose: If True (default), prints progress messages.

    :returns tuple: List of parameter names, list of input parameter values, and 
        array of estimated standard deviations in best-fit parameters.
    """
    params, param_names, model_args = _set_up_model_args(
        graph_file,
        param_file,
        pop_ids=pop_ids,
        bins=bins,
        u=u,
        fitted_u=fitted_u,
        approx_method=approx_method
    )

    if method == "fisher":
        HH = _get_godambe(
            params,
            model_func,
            model_args,
            means,
            varcovs,
            delta=delta,
            return_hessian=True
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(HH)))

    elif method == "godambe":
        if bootstrap_reps is None:
            raise ValueError('Need `bootstrap_reps` to use `godambe` method!')
        GIM, HH, JJ = _get_godambe(
            params,
            model_func,
            model_args,
            means,
            varcovs,
            bootstrap_reps=bootstrap_reps,
            delta=delta,
            return_hessian=False,
            verbose=verbose
        )
        uncerts = np.sqrt(np.diag(np.linalg.inv(GIM)))
    else:
        raise ValueError("Invalid `method`")
    return param_names, params, uncerts


def compute_lrt_adjustment(
    p0,
    model_args,
    means,
    varcovs,
    bootstrap_reps,
    nested_idx, 
    delta=0.01,
    verbose=True
):
    """
    Compute an adjustment for the likelihood-ratio test statistic when 
    likelihoods are composite, after Coffman et al 2015. There is a typo in 
    the supplement indicating that D_adj = D / factor; instead we should use
    D_adj = D * factor.

    D_adj = 2 * (ll(full) - ll(nested)) * factor

    :param array p0: ML parameter values fitted for "simple" (nested) model, 
        extended to include values for "full" model.
    :param tuple model_args: Modul arguments for "complex" model, constructed
        using `_set_up_model_args`.
    :param list means: List of empirical D+ means.
    :param list varcovs: List of covariance matrices obtained with bootstrap.
    :param list bootstrap_reps: List of bootstrap replicate means. All 
        replicates in this list are used to compute the adjustment factor.
    :param array nested_idx: Indices of values in `p0` fixed in "simple" model.
    :param float delta: Step size for finite-differences estimation of first
        and second derivatives.
    :param bool verbose: If True, print progress messages as the Hessian matrix
        is estimated.

    :returns float: Adjustment factor for the LRT.
    """
    d = len(nested_idx)

    def _diff_func(dparams, args=None):
        params = np.array(p0, copy=True)
        params[nested_idx] = dparams
        return _model_func(params, args=model_args)
    
    p_nested = p0[nested_idx]
    GIM, HH, JJ = _get_godambe( 
        p_nested,
        _diff_func,
        model_args,
        means,
        varcovs,
        bootstrap_reps=bootstrap_reps,
        delta=delta,
        verbose=verbose
    )

    factor = np.trace(np.matmul(JJ, np.linalg.inv(HH))) / d
    return factor


_model_cache = dict()


def _get_godambe(
    p0,
    model_func,
    model_args,
    means,
    varcovs,
    bootstrap_reps=None,
    delta=0.01,
    return_hessian=False,
    verbose=True
):
    """
    Compute the Godambe information matrix (GIM). 

    :param array p0: Array of maximum-likelihood parameter values.
    :param function model_func: Function for computing model expectations.
    :param tuple model_args: Arguments for `model_func`
    :param list means: Empirical means
    :param list varcovs: Covariance arrays obtained via bootstrap
    :param list bootstrap_reps: List of means obtained via bootstrap.
    :param float delta: Optional finite differences step size (default 0.01)
    :param bool return_hessian: If True (default False), only estimate the 
        Hessian matrix; this operation does not require `bootstrap_reps`.
    :param bool verbose: If True (default), print updates at the evaluation of 
        each element.

    :returns tuple: Godambe (GIM), Hessian, and variability matrices.
    """
    if bootstrap_reps is None and not return_hessian:
        raise ValueError("You must provide bootstrap replicates")

    def obj_func(params, means, varcovs, model_args):
        key = tuple(params)
        if key in _model_cache:
            model = _model_cache[key]
        else:
            model = model_func(params, model_args)
            _model_cache[key] = model
        return inference.composite_ll(model, means, varcovs)

    HH = -_get_hessian(
        p0, 
        obj_func, 
        model_args,
        means,
        varcovs,
        delta=delta,
        verbose=verbose
    )
    if return_hessian:
        return HH

    # Esimate the "variability matrix" J by averaging U * U^T across boot. reps
    JJ = np.zeros((len(p0), len(p0)))
    for i, rep_means in enumerate(bootstrap_reps):
        cU = _get_score(
            p0, 
            obj_func, 
            model_args,
            rep_means,
            varcovs,
            delta=delta
        )
        cJ = np.matmul(cU, cU.T)
        JJ += cJ
    JJ = JJ / len(bootstrap_reps)
    GIM = np.matmul(np.matmul(HH, np.linalg.inv(JJ)), HH)
    return GIM, HH, JJ


def _get_hessian(
    p0, 
    obj_func, 
    model_args, 
    means, 
    varcovs, 
    delta=0.01,
    verbose=True
):
    """
    Compute the approximate Hessian matrix of the log-likelihood function with 
    finite differences. Uses empirical means and variances/coveriances obtained 
    by bootstrapping. 

    :param array p0: Array of maximum-likelihood parameter values
    :param function obj_func: Function for computing log likelihood
    :param tuple model_args: Arguments for `obj_func`
    :param list means: Empirical means
    :param list varcovs: Covariance arrays obtained via bootstrap
    :param float delta: Optional finite differences step size (default 0.01)
    :param bool verbose: If True (default), print updates at the evaluation of 
        each element.

    :returns array: Estimated Hessian matrix
    """
    hs = delta * p0
    if np.any(hs == 0):
        hs[hs == 0] = delta
    HH = np.zeros((len(p0), len(p0)), dtype=np.float64)
    args = (means, varcovs, model_args)
    f0 = obj_func(p0, *args)
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            if ii == jj:
                vi = np.zeros(len(p0))
                vi[ii] = 1
                if p0[ii] == 0:
                    ff = obj_func(p0 + hs * vi, *args)
                    f2f = obj_func(p0 + hs * 2 * vi, *args)
                    HH[ii, ii] = (f0 - 2 * ff + f2f)
                else:
                    fb = obj_func(p0 + hs * -vi, *args)
                    ff = obj_func(p0 + hs * vi, *args)
                    HH[ii, ii] = (ff - 2 * f0 + fb) / hs[ii] ** 2
            else:
                vi = np.zeros(len(p0))
                vi[ii] = 1
                vj = np.zeros(len(p0))
                vj[jj] = 1
                if p0[ii] == 0 or p0[jj] == 0:
                    fff = obj_func(p0 + hs * (vi + vj), *args)
                    ff0 = obj_func(p0 + hs * vi, *args)
                    f0f = obj_func(p0 + hs * vj, *args)
                    HH[ii, jj] = (fff - ff0 - f0f + f0) / (hs[ii] * hs[jj])
                else:
                    fff = obj_func(p0 + hs * (vi + vj), *args)
                    ffb = obj_func(p0 + hs * (vi - vj), *args)
                    fbf = obj_func(p0 + hs * (-vi + vj), *args)
                    fbb = obj_func(p0 + hs * -(vi + vj), *args)
                    HH[ii, jj] = (fff - ffb - fbf + fbb) / (4 * hs[ii] * hs[jj])
                HH[jj, ii] = HH[ii, jj]
            if verbose:
                print(inference._current_time(), 
                    f"Evaluated Hessian element ({ii}, {jj})")
    return HH


def _get_score(p0, obj_func, model_args, means, varcovs, delta=0.01):
    """
    Compute the score function using finite differences.

    :param array p0: Array of maximum-likelihood parameter values
    :param function obj_func: Function for computing log likelihood
    :param tuple model_args: Arguments for `obj_func`
    :param list means: Empirical means
    :param list varcovs: Covariance arrays obtained via bootstrap
    :param float delta: Optional finite differences step size (default 0.01)

    :returns array: Gradient (score) vector
    """
    hs = delta * p0
    if np.any(hs == 0):
        hs[hs == 0] = delta
    args = (means, varcovs, model_args)
    score = np.zeros((len(p0), 1))
    for ii in range(len(p0)):
        vec = np.zeros(len(p0))
        vec[ii] = 1
        if p0[ii] == 0:
            # One-sided (forward) finite differences
            f0 = obj_func(p0, *args)
            ff = obj_func(p0 + hs * vec, *args)
            score[ii] = (ff - f0) / hs[ii]
        else:
            fb = obj_func(p0 + hs * -vec, *args)
            ff = obj_func(p0 + hs * vec, *args)
            score[ii] = (ff - fb) / (2 * hs[ii])
    return score

