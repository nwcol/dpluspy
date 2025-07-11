"""
Functions for estimating confidence intervals and performing statistical tests
"""

import demes
import numpy as np
import moments

from . import inference


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


def set_up_model_args(
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


def set_up_bounds(param_file, params, param_names=None):
    """
    Set up arrays of upper and lower bounds for parameters using the lower/upper 
    limits and constraints specified in the `options` object loaded by 
    `moments.Demes.Inference._get_params_dict()` from `param_file`.

    :param dict param_file: Pathname of YAML file defining parameters to be 
        loaded with `moments.Demes.Inference._get_params_dict()`.
    :param arr params: Array of ML parameter values.
    :param list param_names: Optional list of parameters for which to construct
        bounds; defaults to all parameters.

    :return tuple: Arrays of lower and upper bounds on parameters.
    """
    options = moments.Demes.Inference._get_params_dict(param_file)
    if param_names is None:
        param_names = [defs["name"] for defs in options["parameters"]]
    all_params = [specs["name"] for specs in options["parameters"]]
    lower = np.zeros(len(all_params), np.float64)
    upper = np.zeros(len(all_params), np.float64)
    for ii, defs in enumerate(options["parameters"]):
        if "lower_bound" in defs:
            lower[ii] = defs["lower_bound"]
        if "upper_bound" in defs:
            upper[ii] = defs["upper_bound"]
        else:
            upper[ii] = np.inf

    if "constraints" in options:
        for defs in options["constraints"]:
            idx0 = all_params.index(defs["params"][0])
            idx1 = all_params.index(defs["params"][1])
            constraint = defs["constraint"]
            if constraint == "greater_than":
                lower[idx0] = max(lower[idx0], params[idx1])
                upper[idx1] = min(upper[idx1], params[idx0])
            elif constraint == "less_than":
                upper[idx0] = min(upper[idx0], params[idx1])
                lower[idx1] = max(lower[idx1], params[idx0])
            else:
                raise ValueError("Invalid constraint")
            
    # Subset to `param_names`
    if len(param_names) != len(all_params):
        idxs = np.array([all_params.index(name) for name in param_names])
        lower = lower[idxs]
        upper = upper[idxs]
    return lower, upper


def FIM_uncerts(
    graph_file,
    param_file,
    means,
    varcovs,
    pop_ids=None,
    bins=None,
    u=None,
    fitted_u=None,
    delta=0.01,
    approx_method=None,
    model_func=_model_func,
    verbose=True,
    bounds=None,
    return_FIM=False
):
    """
    Compute estimated parameter uncertainties using the Fisher Information
    Matrix (FIM). Because our likelihoods are composite and because genetic
    linkage violates statistical assumptions of independence, variances
    estimated in this way will tend to be underestimates.

    :param str graph_file: Pathname of best-fit YAML model file.
    :param str param_file: Pathname of YAML parameter definition file.
    :param list means: Empirical means.
    :param list varcovs: Covariance matrices estimated via bootstrap
    :param list pop_ids: List of populations represented in `means`, 
        corresponding to deme names in `graph_file`.
    :param np.ndarray bins: Array of bin edges.
    :param float u: Fixed mutation rate parameter.
    :param float fitted_u: Mutation rate, if fitted as a free parameter.
    :param float delta: Step size for finite differences evalutation of
        derivatives (default 0.01).
    :param str approx_method: Method to use for approximating bin D+. Choose
        from "midpoint" and "simpsons"; the latter is more accurate but slower
    :param bool verbose: If True, print updates as matrices are computed.
    :param tuple bounds: Optional tuple of arrays defining upper and lower 
        bounds on parameters; if not given, bounds are set up using bounds
        and constraints defined in `param_file`.
    :param bool return_FIM: If True, return the FIM as well as uncerts. 

    :returns tuple: List of parameter names, array of MLE values, array of
        estimated standard errors. 
    """
    params, param_names, model_args = set_up_model_args(
        graph_file,
        param_file,
        pop_ids=pop_ids,
        bins=bins,
        u=u,
        fitted_u=fitted_u,
        approx_method=approx_method
    )
    
    HH = get_godambe(
        params,
        model_func,
        model_args,
        means,
        varcovs,
        just_hess=True,
        delta=delta,
        verbose=verbose,
        bounds=bounds
    )

    uncerts = np.sqrt(np.diag(np.linalg.inv(HH)))
    if return_FIM:
        ret = (param_names, params, uncerts, HH)
    else:
        ret = (param_names, params, uncerts)
    return ret


def GIM_uncerts(
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
    approx_method="simpsons",
    model_func=_model_func,
    verbose=True,
    bounds=None,
    return_GIM=False
):
    """
    Compute estimated parameter uncertainties using the Godambe Information
    Matrix (GIM) to estimate a covariance matrix.

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
    :param list bootstrap_reps: List of bootstrap replicate means. 
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
    :param bool return_GIM: If True (default False), return the GIM and FIM.

    :returns tuple: List of parameter names, list of input parameter values, and 
        array of estimated standard deviations in best-fit parameters.
    """
    params, param_names, model_args = set_up_model_args(
        graph_file,
        param_file,
        pop_ids=pop_ids,
        bins=bins,
        u=u,
        fitted_u=fitted_u,
        approx_method=approx_method
    )

    GIM, HH, JJ, score = get_godambe(
        params,
        model_func,
        model_args,
        means,
        varcovs,
        bootstrap_reps=bootstrap_reps,
        delta=delta,
        verbose=verbose,
        bounds=bounds
    )

    GIM_inv = np.linalg.inv(GIM)
    uncerts = np.sqrt(np.diag(GIM_inv))
    if return_GIM: 
        ret = (param_names, params, uncerts, GIM, HH)
    else:
        ret = (param_names, params, uncerts)
    return ret


def LRT_adjust(
    p0,
    model_args,
    means,
    varcovs,
    bootstrap_reps,
    nested_idx, 
    delta=0.01,
    steps=None,
    verbose=True,
    bounds=None,
    model_func=_model_func,
    return_GIM=False
):
    """
    Compute an adjustment for the likelihood ratio test with composite 
    likelihoods, after Coffman et al 2015. If ll_full and ll_nested are log-
    likelihoods of a complex and nested model, the adjusted statistic is
    `D_adj = adj * 2 * (ll_full - ll_nested)`.

    :param np.ndarray p0: Vector of parameters corresponding to the complex 
        model. Parameters left free in the nested model should be set to nested 
        MLE values, while parameters fixed in the nested model should be set
        to their fixed values. 
    :param tuple model_args: Modul arguments for "complex" model, constructed
        using `_set_up_model_args`.
    :param list means: List of empirical D+ means.
    :param list varcovs: List of covariance matrices obtained with bootstrap.
    :param list bootstrap_reps: List of bootstrap replicate means. All 
        replicates in this list are used to compute the adjustment factor.
    :param np.ndarray nested_idx: Indices of nested parameters.
    :param float delta: Step size for finite-differences estimation of first
        and second derivatives.
    :param bool verbose: If True, print progress messages as the Hessian matrix
        is estimated.
    :param tuple bounds: Optional tuple of lower and upper bounds on parameters.
    :param function model_func: Function to evaluate.
    :param bool return_GIM: If True (default False), return the GIM, sensitivity
        and variability matrices

    :returns float: Adjustment factor for the LRT.
    """
    nested_idx = np.asarray(nested_idx)

    def pass_func(dparams, args=None):
        params = np.array(p0, copy=True)
        params[nested_idx] = dparams
        return model_func(params, args=model_args)
    
    p_nested = p0[nested_idx]
    GIM, HH, JJ, score = get_godambe( 
        p_nested,
        pass_func,
        model_args,
        means,
        varcovs,
        bootstrap_reps=bootstrap_reps,
        delta=delta,
        steps=steps,
        verbose=verbose,
        bounds=bounds
    )
    factor = len(nested_idx) / np.trace(np.matmul(JJ, np.linalg.inv(HH)))
    if return_GIM:
        ret = (factor, GIM, HH, JJ)
    else: 
        ret = factor
    return ret


_model_cache = dict()


def get_godambe(
    p0,
    model_func,
    model_args,
    means,
    varcovs,
    bootstrap_reps=None,
    delta=0.01,
    steps=None,
    just_hess=False,
    verbose=True,
    bounds=None
):
    """
    Compute the Godambe information matrix (GIM), which is `G = H J^-1 H`
    where `H` is the sensitivity matrix and `J` is the variability matrix 
    (the expected variance of the score function, estimated by bootstrap).

    :param array p0: Array of maximum-likelihood parameter values.
    :param function model_func: Function for computing model expectations.
    :param tuple model_args: Arguments for `model_func`. See `set_up_model_args`
    :param list means: Empirical means
    :param list varcovs: Covariance arrays obtained via bootstrap
    :param list bootstrap_reps: List of means obtained via bootstrap. Not 
        required if `just_hess` is True.
    :param float delta: Optional finite differences step size (default 0.01)
    :param bool just_hess: If True (default False), only estimate the 
        Hessian matrix; this operation does not require `bootstrap_reps`.
    :param bool verbose: If True (default), print updates at the evaluation of 
        each element.
    :param tuple bounds: Optional tuple of upper and lower bounds on parameters; 
        each should have the same length as `p0`. We use one-sided finite 
        differences to evaluate the derivative for parameters within `delta`
        times their value of a bound. It is assumed that bounds are inclusive;
        that the objective function can be evaluated at the value of the bound, 
        unless it is an infinite upper bound.

    :returns tuple: Godambe Information, sensitivity, and variability matrices
        and expected score.
    """
    if bootstrap_reps is None and not just_hess:
        raise ValueError("You must provide bootstrap replicates")

    def obj_func(params, means, varcovs, model_args):
        key = tuple(params)
        if key in _model_cache:
            model = _model_cache[key]
        else:
            model = model_func(params, model_args)
            _model_cache[key] = model
        return inference.composite_ll(model, means, varcovs)

    HH = -get_hess(
        p0, 
        obj_func, 
        model_args,
        means,
        varcovs,
        delta=delta,
        steps=steps,
        verbose=verbose,
        bounds=bounds
    )
    if just_hess:
        return HH

    # Compute expected score and J across bootstrap realizations
    score = np.zeros((len(p0), 1))
    JJ = np.zeros((len(p0), len(p0)))
    for rep_means in bootstrap_reps:
        cU = get_grad(
            p0, 
            obj_func, 
            model_args,
            rep_means,
            varcovs,
            delta=delta,
            steps=steps,
            bounds=bounds
        )
        score += cU
        _J = np.matmul(cU, cU.T)
        JJ += _J
    score = score / len(bootstrap_reps)
    JJ = JJ / len(bootstrap_reps)
    J_inv = np.linalg.inv(JJ)
    GIM = HH @ J_inv @ HH
    return GIM, HH, JJ, score


def get_hess(
    p0, 
    func, 
    model_args, 
    means, 
    varcovs, 
    delta=0.01,
    steps=None,
    verbose=True,
    bounds=None
):
    """
    Compute the approximate Hessian matrix of the log-likelihood function using 
    finite differences. Uses empirical means and variances/coveriances obtained 
    by bootstrapping. 

    :param array p0: Array of maximum-likelihood parameter values
    :param function func: Function for computing log likelihood
    :param tuple model_args: Arguments for `func`
    :param list means: Empirical means
    :param list varcovs: Covariance arrays obtained via bootstrap
    :param float delta: Optional finite differences step size (default 0.01)
    :param bool verbose: If True (default), print updates at the evaluation of 
        each element.    
    :param tuple bounds: Optional tuple of upper and lower bounds on parameters; 
        each should have the same length as `p0`. We use one-sided finite 
        differences to evaluate the derivative for parameters closer to a bound
        than `delta` times their value. It is assumed that bounds are inclusive;
        that `func` can be evaluated at the value of the bound, unless it is an 
        infinite upper bound.

    :returns array: Estimated Hessian matrix
    """
    # Check bounds; parameters are allowed to equal 0.
    if bounds is None:
        bounds = (np.zeros(len(p0)), np.full(len(p0), np.inf))
    if np.any(p0 < bounds[0]) or np.any(p0 >= bounds[1]):
        raise ValueError("All parameters must be within bounds")
    
    if steps is not None:
        steps = np.asarray(steps)
        if len(steps) != len(p0):
            raise ValueError("Lengths of `steps`, `p0` must be equal")
    else:
        steps = delta * p0
        if np.any(steps == 0):
            steps[steps == 0] = delta

    for ii in range(len(p0)):
        if np.any((p0 - steps <= bounds[0]) & (p0 + steps >= bounds[1])):
            raise ValueError(
                f"Parameter {ii} bounds prevent finite differences evaluation")
        
    args = (means, varcovs, model_args)
    
    HH = np.zeros((len(p0), len(p0)), dtype=np.float64)
    for ii in range(len(p0)):
        for jj in range(ii, len(p0)):
            elem = hessian_elem(func, p0, steps, bounds, ii, jj, args=args)
            if ii == jj:
                HH[ii, jj] = elem
            else:
                HH[ii, jj] = HH[jj, ii] = elem
            if verbose:
                print(inference._current_time(), 
                    f"Evaluated Hessian element ({ii}, {jj})")
    return HH


def hessian_elem(func, p0, steps, bounds, ii, jj, args=()):
    """
    Evaluate element (ii, jj) of the Hessian matrix, the matrix of second
    partial derivatives over the log-likelihood function with respect to 
    parameters ii and jj.

    :param function func: Objective function, taking a vector of parameters
        and `args` as arguments and returning a log-likelihood.
    :param float p0: ML parameter values.
    :param np.ndarray steps: Array of steps to use for each parameter in 
        finite difference evaluation.
    :param tuple bounds: Tuple of arrays of lower and upper parameter bounds.
    :params int ii, jj: Indices of the Hessian element to evaluate.
    :param tuple args: Auxiliary arguments for `func`.

    :returns float: Hessian element (ii, jj)
    """
    indicator = lambda n, i: np.array([0 if j != i else 1 for j in range(n)])
    lower, upper = bounds
    f_0 = func(p0, *args)

    if ii == jj:
        Ii = indicator(len(p0), ii)
        # Forward
        if p0[ii] - steps[ii] <= lower[ii]:
            f_f = func(p0 + steps * Ii, *args)
            f_2f = func(p0 + 2 * steps * Ii, *args)
            elem = (f_0 - 2 * f_f + f_2f) / steps[ii] ** 2
        # Backward
        elif p0[ii] + steps[ii] >= upper[ii]:
            f_b = func(p0 - steps * Ii, *args)
            f_2b = func(p0 - 2 * steps * Ii, *args)
            elem = (f_0 - 2 * f_b + f_2b) / steps[ii] ** 2
        # Central
        else:
            f_b = func(p0 - steps * Ii, *args)
            f_f = func(p0 + steps * Ii, *args)
            elem = (f_f - 2 * f_0 + f_b) / steps[ii] ** 2

    else:
        Ii = indicator(len(p0), ii)
        Ij = indicator(len(p0), jj)

        if p0[ii] + steps[ii] >= upper[ii]:
            # Backward/backward
            if p0[jj] + steps[jj] >= upper[jj]: 
                f_bb = func(p0 - steps * (Ii + Ij), *args)
                f_0b = func(p0 - steps * Ij, *args)
                f_b0 = func(p0 - steps * Ii, *args)
                elem = (f_bb - f_0b - f_b0 + f_0) / (steps[ii] * steps[jj])
            # Backward/forward
            elif p0[jj] - steps[jj] <= lower[jj]:
                f_0f = func(p0 + steps * Ij, *args)
                f_bf = func(p0 + steps * (Ij - Ii), *args)
                f_b0 = func(p0 - steps * Ii, *args)
                elem = (f_0f - f_0 - f_bf + f_b0) / (steps[ii] * steps[jj])
            # Backward/central
            else:
                f_0f = func(p0 + steps * Ij, *args)
                f_bf = func(p0 + steps * (Ij - Ii), *args)
                f_0b = func(p0 - steps * Ij, *args)
                f_bb = func(p0 - steps * (Ii + Ij), *args)
                elem = (f_0f - f_bf - f_0b + f_bb) / (2 * steps[ii] * steps[jj])

        elif p0[ii] - steps[ii] <= lower[ii]:
            # Forward/backward
            if p0[jj] + steps[jj] >= upper[jj]: 
                f_f0 = func(p0 + steps * Ii, *args)
                f_fb = func(p0 + steps * (Ii - Ij), *args)
                f_0b = func(p0 - steps * Ij, *args)
                elem = (f_f0 - f_fb - f_0 + f_0b) / (steps[ii] * steps[jj])
            # Forward/forward
            elif p0[jj] - steps[jj] <= lower[jj]:
                f_ff = func(p0 + steps * (Ii + Ij), *args)
                f_f0 = func(p0 + steps * Ii, *args)
                f_0f = func(p0 + steps * Ij, *args)
                elem = (f_ff - f_f0 - f_0f + f_0) / (steps[ii] * steps[jj])
            # Forward/central
            else:
                f_ff = func(p0 + steps * (Ii + Ij), *args)
                f_fb = func(p0 + steps * (Ii - Ij), *args)
                f_0f = func(p0 + steps * Ij, *args)
                f_0b = func(p0 - steps * Ij, *args)
                elem = (f_ff - f_fb - f_0f + f_0b) / (2 * steps[ii] * steps[jj])

        else:
            # Central/backward
            if p0[jj] + steps[jj] >= upper[jj]: 
                f_f0 = func(p0 + steps * Ii, *args)
                f_fb = func(p0 + steps * (Ii - Ij), *args)
                f_b0 = func(p0 - steps * Ii, *args)
                f_bb = func(p0 - steps * (Ii + Ij), *args)
                elem = (f_f0 - f_fb - f_b0 + f_bb) / (2 * steps[ii] * steps[jj])
            # Central/forward
            elif p0[jj] - steps[jj] <= lower[jj]:
                f_ff = func(p0 + steps * (Ii + Ij), *args)
                f_f0 = func(p0 + steps * Ii, *args)
                f_bf = func(p0 + steps * (Ij - Ii), *args)
                f_b0 = func(p0 - steps * Ii, *args)
                elem = (f_ff - f_f0 - f_bf + f_b0) / (2 * steps[ii] * steps[jj])
            # Central/central
            else:
                f_ff = func(p0 + steps * (Ii + Ij), *args)
                f_fb = func(p0 + steps * (Ii - Ij), *args)
                f_bf = func(p0 + steps * (Ij - Ii), *args)
                f_bb = func(p0 - steps * (Ii + Ij), *args)
                elem = (f_ff - f_fb - f_bf + f_bb) / (4 * steps[ii] * steps[jj])

    return elem


def get_grad(
    p0, 
    func, 
    model_args, 
    means, 
    varcovs, 
    delta=0.01, 
    steps=None,
    bounds=None
):
    """
    Compute the score function using finite differences.

    :param array p0: Array of maximum-likelihood parameter values
    :param function func: Function for computing log likelihood
    :param tuple model_args: Arguments for `func`
    :param list means: Empirical means
    :param list varcovs: Covariance arrays obtained via bootstrap
    :param float delta: Optional finite differences step size (default 0.01)
    :param tuple bounds: Optional tuple of upper and lower bounds on parameters; 
        each should have the same length as `p0`. We use one-sided finite 
        differences to evaluate the derivative for parameters closer to a bound
        than `delta` times their value. It is assumed that bounds are inclusive;
        that `func` can be evaluated at the value of the bound, unless it
        is an infinite upper bound.

    :returns array: Gradient (score) vector
    """
    if bounds is None:
        bounds = (np.zeros(len(p0)), np.full(len(p0), np.inf))
    if np.any(p0 < bounds[0]) or np.any(p0 > bounds[1]):
        raise ValueError("All parameters must be within bounds")

    if steps is not None:
        steps = np.asarray(steps)
        if len(steps) != len(p0):
            raise ValueError("Lengths of `steps`, `p0` must be equal")
    else:
        steps = delta * p0
        if np.any(steps == 0):
            steps[steps == 0] = delta

    for ii in range(len(p0)):
        if np.any((p0 - steps <= bounds[0]) & (p0 + steps >= bounds[1])):
            raise ValueError(
                f"Parameter {ii} bounds prevent finite differences evaluation")
        
    indicator = lambda n, i: np.array([0 if j != i else 1 for j in range(n)])
    args = (means, varcovs, model_args)
    score = np.zeros((len(p0), 1))
    for ii in range(len(p0)):
        Ii = indicator(len(p0), ii)
        # One-sided (forward) finite differences
        if p0[ii] - steps[ii] <= bounds[0][ii]:
            f_0 = func(p0, *args)
            f_f = func(p0 + steps * Ii, *args)
            score[ii, 0] = (f_f - f_0) / steps[ii]

        # One-sided (backward) finite differences
        elif p0[ii] + steps[ii] >= bounds[1][ii]:
            f_0 = func(p0, *args)
            f_b = func(p0 - steps * Ii, *args)
            score[ii, 0] = (f_0 - f_b) / steps[ii]

        # Central
        else:
            f_b = func(p0 - steps * Ii, *args)
            f_f = func(p0 + steps * Ii, *args)
            score[ii, 0] = (f_f - f_b) / (2 * steps[ii])

    return score

