"""Effective binomial sample size of a cause-specific death rate."""

import pandas as pd

# Normal-consistency factor from median absolute deviation to SD.
_MAD_TO_SD = 1.4826


def vr_effective_sample_size(
    cause_fraction: pd.Series,
    all_cause_death_rate: pd.Series,
    all_cause_death_rate_sd: pd.Series,
    population: pd.Series,
    completeness: pd.Series,
    pct_garbage: pd.Series,
    logit_pct_garbage_sd: pd.Series,
    envelope_ub: float,
) -> pd.Series:
    """VR effective binomial sample size."""
    n = completeness * population
    a = all_cause_death_rate * (
        envelope_ub - cause_fraction * all_cause_death_rate
    )
    b = cause_fraction * (
        all_cause_death_rate_sd**2
        + pct_garbage**2
        * logit_pct_garbage_sd**2
        * (all_cause_death_rate**2 + all_cause_death_rate_sd**2)
    )
    numerator = n * a
    denominator = a + (n - 1.0).clip(lower=0.0) * b
    return numerator / denominator


def nonvr_effective_sample_size(
    cause_fraction: pd.Series,
    sample_size: pd.Series,
    all_cause_death_rate: pd.Series,
    all_cause_death_rate_sd: pd.Series,
    envelope: pd.Series,
    pct_garbage: pd.Series,
    logit_pct_garbage_sd: pd.Series,
    envelope_ub: float,
) -> pd.Series:
    """VA effective binomial sample size."""
    # A cause's VA count can't exceed the all-cause envelope; capping there also keeps the
    # effective all-cause population env_nonvr / all_cause_death_rate <= n_pop, no separate population cap is needed.
    env_nonvr = sample_size.clip(upper=envelope)
    pop_nonvr = env_nonvr / all_cause_death_rate
    # All-cause variance on the effective population implied by the VA sample.
    var_ra_nonvr = (
        all_cause_death_rate * (envelope_ub - all_cause_death_rate) / pop_nonvr
        + (pop_nonvr - 1.0).clip(lower=0.0)
        / pop_nonvr
        * all_cause_death_rate_sd**2
    )
    v = var_ra_nonvr + all_cause_death_rate**2
    a = (envelope_ub - cause_fraction) * v
    b = (
        (env_nonvr - 1.0).clip(lower=0.0)
        * cause_fraction
        * pct_garbage**2
        * logit_pct_garbage_sd**2
        * v
    )
    c = env_nonvr * cause_fraction * var_ra_nonvr
    numerator = (
        env_nonvr
        * all_cause_death_rate
        * (envelope_ub - cause_fraction * all_cause_death_rate)
    )
    denominator = a + b + c
    return numerator / denominator


def effective_sample_size(
    is_vr: pd.Series,
    cause_fraction: pd.Series,
    sample_size: pd.Series,
    population: pd.Series,
    envelope: pd.Series,
    envelope_sd: pd.Series,
    completeness: pd.Series,
    pct_garbage: pd.Series,
    logit_pct_garbage_mad: pd.Series,
    envelope_ub: float,
) -> pd.Series:
    """Per-row effective binomial sample size: VR where ``is_vr`` else VA."""
    # Compute the all-cause death rate and convert SD from death-count to rate scale
    all_cause_death_rate = envelope / population
    all_cause_death_rate_sd = envelope_sd / population
    # Derive the redistribution SD from the logit-pct-garbage MAD (source is a MAD)
    logit_pct_garbage_sd = _MAD_TO_SD * logit_pct_garbage_mad

    weights = pd.Series(index=is_vr.index, dtype="float64")
    weights[is_vr] = vr_effective_sample_size(
        cause_fraction=cause_fraction[is_vr],
        all_cause_death_rate=all_cause_death_rate[is_vr],
        all_cause_death_rate_sd=all_cause_death_rate_sd[is_vr],
        population=population[is_vr],
        completeness=completeness[is_vr],
        pct_garbage=pct_garbage[is_vr],
        logit_pct_garbage_sd=logit_pct_garbage_sd[is_vr],
        envelope_ub=envelope_ub,
    )
    weights[~is_vr] = nonvr_effective_sample_size(
        cause_fraction=cause_fraction[~is_vr],
        sample_size=sample_size[~is_vr],
        all_cause_death_rate=all_cause_death_rate[~is_vr],
        all_cause_death_rate_sd=all_cause_death_rate_sd[~is_vr],
        envelope=envelope[~is_vr],
        pct_garbage=pct_garbage[~is_vr],
        logit_pct_garbage_sd=logit_pct_garbage_sd[~is_vr],
        envelope_ub=envelope_ub,
    )
    return weights
