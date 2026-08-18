"""Effective binomial sample size of a cause-specific death rate."""

import numpy as np
import pandas as pd

# Normal-consistency factor from median absolute deviation to SD.
_MAD_TO_SD = 1.4826


def _vr_effective_sample_size(
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


def _non_vr_effective_sample_size(
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


def _validate_data(
    data: pd.DataFrame,
    is_vr: str,
    cause_fraction: str,
    sample_size: str,
    population: str,
    envelope: str,
    envelope_sd: str,
    completeness: str,
    pct_garbage: str,
    logit_pct_garbage_mad: str,
    envelope_ub: float,
) -> None:
    nona_columns = [
        is_vr,
        cause_fraction,
        sample_size,
        population,
        envelope,
        envelope_sd,
        pct_garbage,
        logit_pct_garbage_mad,
    ]
    na_cols = [col for col in nona_columns if data[col].isna().any()]
    if len(na_cols) > 0:
        raise ValueError(f"Columns contain missing values: {na_cols}")

    if not pd.api.types.is_bool_dtype(data[is_vr]):
        raise ValueError(f"{is_vr} must be in boolean type")

    vr_mask = data[is_vr].to_numpy()
    if not data.loc[vr_mask, completeness].between(0.0, 1.0).all():
        raise ValueError(f"{completeness} must be in [0, 1] for VR rows")

    for col in (cause_fraction, pct_garbage):
        if not data[col].between(0.0, 1.0).all():
            raise ValueError(f"{col} must be in [0, 1]")

    for col in (sample_size, population, envelope, envelope_sd):
        if (data[col] <= 0).any():
            raise ValueError(f"{col} must be positive")

    if (data[envelope] / data[population] > envelope_ub).any():
        raise ValueError(
            f"{envelope} / {population} must be less than or equal to envelope_ub ({envelope_ub})"
        )


def effective_sample_size(
    data: pd.DataFrame,
    is_vr: str,
    cause_fraction: str,
    sample_size: str,
    population: str,
    envelope: str,
    envelope_sd: str,
    completeness: str,
    pct_garbage: str,
    logit_pct_garbage_mad: str,
    envelope_ub: float,
) -> pd.Series:
    """Per-row effective binomial sample size: VR where ``is_vr`` else VA."""
    _validate_data(
        data,
        is_vr,
        cause_fraction,
        sample_size,
        population,
        envelope,
        envelope_sd,
        completeness,
        pct_garbage,
        logit_pct_garbage_mad,
        envelope_ub,
    )

    # Compute the all-cause death rate and convert SD from death-count to rate scale
    all_cause_death_rate = data[envelope] / data[population]
    all_cause_death_rate_sd = data[envelope_sd] / data[population]
    # Derive the redistribution SD from the logit-pct-garbage MAD (source is a MAD)
    logit_pct_garbage_sd = _MAD_TO_SD * data[logit_pct_garbage_mad]

    weights = pd.Series(index=data.index, dtype="float64")
    vr_mask = data[is_vr].to_numpy()
    weights[vr_mask] = _vr_effective_sample_size(
        cause_fraction=data.loc[vr_mask, cause_fraction],
        all_cause_death_rate=all_cause_death_rate[vr_mask],
        all_cause_death_rate_sd=all_cause_death_rate_sd[vr_mask],
        population=data.loc[vr_mask, population],
        completeness=data.loc[vr_mask, completeness],
        pct_garbage=data.loc[vr_mask, pct_garbage],
        logit_pct_garbage_sd=logit_pct_garbage_sd[vr_mask],
        envelope_ub=envelope_ub,
    )
    weights[~vr_mask] = _non_vr_effective_sample_size(
        cause_fraction=data.loc[~vr_mask, cause_fraction],
        sample_size=data.loc[~vr_mask, sample_size],
        all_cause_death_rate=all_cause_death_rate[~vr_mask],
        all_cause_death_rate_sd=all_cause_death_rate_sd[~vr_mask],
        envelope=data.loc[~vr_mask, envelope],
        pct_garbage=data.loc[~vr_mask, pct_garbage],
        logit_pct_garbage_sd=logit_pct_garbage_sd[~vr_mask],
        envelope_ub=envelope_ub,
    )

    not_valid = ~np.isfinite(weights) | (weights < 0)
    if not_valid.any():
        raise ValueError(
            f"Invalid effective sample size for rows: {list(data.index[not_valid])}"
        )
    return weights
