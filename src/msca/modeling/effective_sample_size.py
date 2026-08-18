"""Effective binomial sample size of a cause-specific death rate."""

import pandas as pd

# Normal-consistency factor from median absolute deviation to SD.
_MAD_TO_SD = 1.4826


def vr_effective_sample_size(
    completeness: pd.Series,
    population: pd.Series,
    mx: pd.Series,
    cause_fraction: pd.Series,
    mx_sd: pd.Series,
    pct_garbage: pd.Series,
    logit_pct_garbage_sd: pd.Series,
    envelope_ub: float,
) -> pd.Series:
    """VR effective binomial sample size."""
    n = completeness * population
    a = mx * (envelope_ub - cause_fraction * mx)
    b = cause_fraction * (
        mx_sd**2 + pct_garbage**2 * logit_pct_garbage_sd**2 * (mx**2 + mx_sd**2)
    )
    numerator = n * a
    denominator = a + (n - 1.0).clip(lower=0.0) * b
    return numerator / denominator


def va_effective_sample_size(
    sample_size: pd.Series,
    envelope: pd.Series,
    mx: pd.Series,
    cause_fraction: pd.Series,
    mx_sd: pd.Series,
    pct_garbage: pd.Series,
    logit_pct_garbage_sd: pd.Series,
    envelope_ub: float,
) -> pd.Series:
    """VA effective binomial sample size."""
    # A cause's VA count can't exceed the all-cause envelope; capping there also keeps the
    # effective all-cause population env_va / mx <= n_pop, no separate population cap is needed.
    env_va = sample_size.clip(upper=envelope)
    pop_va = env_va / mx
    # All-cause variance on the effective population implied by the VA sample.
    var_ra_va = (
        mx * (envelope_ub - mx) / pop_va
        + (pop_va - 1.0).clip(lower=0.0) / pop_va * mx_sd**2
    )
    v = var_ra_va + mx**2
    a = (envelope_ub - cause_fraction) * v
    b = (
        (env_va - 1.0).clip(lower=0.0)
        * cause_fraction
        * pct_garbage**2
        * logit_pct_garbage_sd**2
        * v
    )
    c = env_va * cause_fraction * var_ra_va
    numerator = env_va * mx * (envelope_ub - cause_fraction * mx)
    denominator = a + b + c
    return numerator / denominator


def effective_sample_size(
    is_vr: pd.Series,
    completeness: pd.Series,
    sample_size: pd.Series,
    population: pd.Series,
    envelope: pd.Series,
    envelope_sd: pd.Series,
    cause_fraction: pd.Series,
    pct_garbage: pd.Series,
    logit_pct_garbage_mad: pd.Series,
    envelope_ub: float,
) -> pd.Series:
    """Per-row effective binomial sample size: VR where ``is_vr`` else VA."""
    # Compute the all-cause death rate and convert SD from death-count to rate scale
    mx = envelope / population
    mx_sd = envelope_sd / population
    # Derive the redistribution SD from the logit-pct-garbage MAD (source is a MAD)
    logit_pct_garbage_sd = _MAD_TO_SD * logit_pct_garbage_mad

    weights = pd.Series(index=is_vr.index, dtype="float64")
    weights[is_vr] = vr_effective_sample_size(
        completeness=completeness[is_vr],
        population=population[is_vr],
        mx=mx[is_vr],
        cause_fraction=cause_fraction[is_vr],
        mx_sd=mx_sd[is_vr],
        pct_garbage=pct_garbage[is_vr],
        logit_pct_garbage_sd=logit_pct_garbage_sd[is_vr],
        envelope_ub=envelope_ub,
    )
    weights[~is_vr] = va_effective_sample_size(
        sample_size=sample_size[~is_vr],
        envelope=envelope[~is_vr],
        mx=mx[~is_vr],
        cause_fraction=cause_fraction[~is_vr],
        mx_sd=mx_sd[~is_vr],
        pct_garbage=pct_garbage[~is_vr],
        logit_pct_garbage_sd=logit_pct_garbage_sd[~is_vr],
        envelope_ub=envelope_ub,
    )
    return weights
