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
    """Effective binomial sample size for vital registration rows.

    Parameters
    ----------
    cause_fraction
        Fraction of all-cause deaths assigned to the cause.
    all_cause_death_rate
        All-cause death rate, ``envelope / population``.
    all_cause_death_rate_sd
        Standard deviation of ``all_cause_death_rate``.
    population
        Population at risk.
    completeness
        Fraction of deaths captured by the registration system. The
        registered death count ``completeness * population`` is the
        upper limit that the variance terms shrink towards.
    pct_garbage
        Fraction of the cause's deaths that came from redistributed
        garbage codes.
    logit_pct_garbage_sd
        Standard deviation of ``pct_garbage`` on the logit scale.
    envelope_ub
        Upper bound of the all-cause death rate. See
        :func:`effective_sample_size`.

    Returns
    -------
    Series
        Effective binomial sample size for each row.

    """
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
    """Effective binomial sample size for non-registration rows.

    Unlike the vital registration form, the population at risk is not
    taken from the data. It is implied by the reported sample size,
    which stands in for the source's all-cause death count.

    Parameters
    ----------
    cause_fraction
        Fraction of all-cause deaths assigned to the cause.
    sample_size
        Deaths reported by the source, capped at ``envelope``.
    all_cause_death_rate
        All-cause death rate, ``envelope / population``.
    all_cause_death_rate_sd
        Standard deviation of ``all_cause_death_rate``.
    envelope
        All-cause deaths, used to cap ``sample_size``.
    pct_garbage
        Fraction of the cause's deaths that came from redistributed
        garbage codes.
    logit_pct_garbage_sd
        Standard deviation of ``pct_garbage`` on the logit scale.
    envelope_ub
        Upper bound of the all-cause death rate. See
        :func:`effective_sample_size`.

    Returns
    -------
    Series
        Effective binomial sample size for each row.

    """
    # A cause's death count can't exceed the all-cause envelope. Capping
    # there also bounds the implied all-cause population, because
    # env_nonvr / all_cause_death_rate <= envelope / all_cause_death_rate,
    # which is the population itself, so no separate population cap is
    # needed.
    env_nonvr = sample_size.clip(upper=envelope)
    pop_nonvr = env_nonvr / all_cause_death_rate
    # All-cause variance on the effective population implied by the sample.
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
    """Check the inputs of :func:`effective_sample_size`.

    Raises
    ------
    ValueError
        If any input is missing, of the wrong dtype, or out of range.

    """
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

    # completeness only feeds the VR form, so it is unconstrained elsewhere
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
    """Effective binomial sample size of a cause-specific death rate.

    The cause-specific death rate is treated as a binomial proportion on
    ``[0, envelope_ub]``. The value returned for a row is the binomial
    sample size whose sampling variance matches that row's total
    variance, which combines uncertainty in the all-cause envelope with
    uncertainty from garbage-code redistribution. It is intended for use
    as an observation weight, and is always at most the row's nominal
    sample size.

    Registration and non-registration rows are computed differently:
    the former take their population at risk from the data, the latter
    infer it from the reported sample size. ``is_vr`` selects between
    them.

    Parameters
    ----------
    data
        Observation rows. Every argument below other than ``envelope_ub``
        names a column to read from it.
    is_vr
        Column flagging vital registration rows. Must be a boolean
        dtype; 0/1 integer columns are rejected rather than coerced.
    cause_fraction
        Column of the fraction of all-cause deaths assigned to the
        cause, in ``[0, 1]``.
    sample_size
        Column of deaths reported by the source. Only the non-VR form
        reads it, where it is capped at ``envelope``, but it must be
        present and positive on every row.
    population
        Column of population at risk. Must be positive.
    envelope
        Column of all-cause deaths. Must be positive.
    envelope_sd
        Column of the standard deviation of ``envelope``. Must be
        positive.
    completeness
        Column of the fraction of deaths captured by the registration
        system, in ``[0, 1]``. Only the VR form reads it, and it is only
        validated on VR rows, so it may be missing on the others.
    pct_garbage
        Column of the fraction of the cause's deaths that came from
        redistributed garbage codes, in ``[0, 1]``.
    logit_pct_garbage_mad
        Column of the median absolute deviation of ``pct_garbage`` on
        the logit scale. Converted to a standard deviation with the
        normal-consistency factor 1.4826.
    envelope_ub
        Upper bound of the all-cause death rate, on the same scale as
        ``envelope / population``, which must not exceed it.

    Returns
    -------
    Series
        Effective binomial sample size for each row of ``data``, sharing
        its index. Zero is a valid result and means the row carries no
        information, as happens when ``completeness`` is zero.

    Raises
    ------
    ValueError
        If a required column has missing values, ``is_vr`` is not a
        boolean dtype, a proportion falls outside ``[0, 1]``, a
        quantity required to be positive is not, ``envelope /
        population`` exceeds ``envelope_ub``, or a computed sample size
        comes out non-finite or negative.

    Notes
    -----
    ``envelope`` and ``envelope_sd`` are given on the death-count scale
    and are converted to rates here by dividing by ``population``.

    Examples
    --------
    >>> weights = effective_sample_size(
    ...     data,
    ...     is_vr="is_vr",
    ...     cause_fraction="cause_fraction",
    ...     sample_size="sample_size",
    ...     population="population",
    ...     envelope="envelope",
    ...     envelope_sd="envelope_sd",
    ...     completeness="completeness",
    ...     pct_garbage="pct_garbage",
    ...     logit_pct_garbage_mad="variance_rd_logit_cf",
    ...     envelope_ub=4.0,
    ... )

    """
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
