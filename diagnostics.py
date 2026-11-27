"""Path diagnostics for the BBM and I-BBM. Time is in trading days, 252 per year.

Input level arrays contain the opening observation and T successive closes.
An observation at index t is the close after trading step t. No artificial
zero return is added, and the warm-up is excluded by the caller.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


DAYS = 252
STAGE_NAMES = ("accumulation", "markup", "distribution", "markdown")


def _log_levels(log_index):
    levels = np.asarray(log_index, dtype=float)
    if levels.ndim != 1 or len(levels) < 2 or not np.isfinite(levels).all():
        raise ValueError("log_index must contain at least two finite log levels")
    return levels


@dataclass(frozen=True)
class Drawdown:
    peak: int
    breach: int
    trough: int
    rebound: Optional[int]
    depth: float


def drawdown_episodes(log_index, threshold=0.20):
    """Date directional-change episodes in log *levels* at a simple-price threshold.

    A drop by threshold from a local high fires once. The next search begins
    after a rally by threshold from the running low, even below the old peak.
    A terminal open episode counts once, with an observed-to-date depth and
    rebound=None. 'breach' is the first detection date. Troughs are retrospective.
    """
    levels = _log_levels(log_index)
    if not 0 < threshold < 1:
        raise ValueError("threshold must be between zero and one")
    fall, rise = np.log1p(-threshold), np.log1p(threshold)
    # Tolerate last-bit log rounding at an exact threshold such as 100 -> 80.
    tolerance = 8 * np.finfo(float).eps * max(1.0, np.max(np.abs(levels)))
    peak, trough = 0, 0
    breach = None
    episodes = []
    for day in range(1, len(levels)):
        value = levels[day]
        if breach is None:
            if value >= levels[peak]:
                peak = day
            elif value - levels[peak] <= fall + tolerance:
                breach, trough = day, day
        elif value < levels[trough]:
            trough = day
        elif value - levels[trough] >= rise - tolerance:
            depth = float(-np.expm1(levels[trough] - levels[peak]))
            episodes.append(Drawdown(peak, breach, trough, day, depth))
            peak, breach = day, None
    if breach is not None:
        depth = float(-np.expm1(levels[trough] - levels[peak]))
        episodes.append(Drawdown(peak, breach, trough, None, depth))
    return episodes


def cycle_stages(log_index):
    """Return retrospective Section 5.1 labels, with -1 for undated observations.

    Stage labels describe the return ENDING on each day. Only complete intervals
    between two dated peaks/troughs are used for bullish stages. A terminal bear
    whose trough has yet to be confirmed is not labelled. A 200-close arithmetic
    price SMA is required for bullish stages; if missing, that interval is left
    undated. Distribution is the contiguous pre-peak segment within 10% of the
    next peak, truncated at the end of accumulation if the criteria overlap.
    """
    levels = _log_levels(log_index)
    events = [event for event in drawdown_episodes(levels) if event.rebound is not None]
    stage = np.full(len(levels), -1, dtype=np.int8)
    for event in events:
        stage[event.peak + 1 : event.trough + 1] = 3
    if len(events) < 2:
        return stage

    # Test S[t] > mean(S[t-199:t+1]) using a local scale for long horizons.
    above = np.zeros(len(levels), dtype=bool)
    window_size = 200
    for start in range(window_size - 1, len(levels), 2048):
        stop = min(start + 2048, len(levels))
        window = np.lib.stride_tricks.sliding_window_view(
            levels[start - window_size + 1 : stop], window_size
        )
        above[start:stop] = (np.exp(window - window[:, -1, None]).mean(axis=1) < 1)
    for previous, following in zip(events, events[1:]):
        start, finish = previous.trough + 1, following.peak
        if start < window_size - 1 or finish < start:
            continue
        crossings = np.flatnonzero(above[start : finish + 1])
        end_accumulation = start + int(crossings[0]) if len(crossings) else finish
        stage[start : end_accumulation + 1] = 0
        bull_start = end_accumulation + 1
        below = np.flatnonzero(levels[bull_start : finish + 1] < levels[finish] + np.log(0.9))
        distribution_start = bull_start + int(below[-1]) + 1 if len(below) else bull_start
        stage[bull_start:distribution_start] = 1
        stage[distribution_start : finish + 1] = 2
    return stage


def _shape(values, power):
    if len(values) < power:
        return np.nan
    centered = values - np.mean(values)
    deviation = np.sqrt(np.mean(centered**2))
    if deviation == 0:
        return np.nan
    return float(np.mean((centered / deviation) ** power) - (3 if power == 4 else 0))


def index_moments(log_index):
    """Compute Table 5/11 index observables and Table 33 cycle descriptors.

    Volatility and distributional moments use log returns. Horizon skew uses
    disjoint blocks anchored at the first observation, dropping the incomplete
    last block. Variance ratios use overlapping h-day differences of log levels.
    Mean bear depth is a positive simple fraction; maximum log drawdown uses the
    all-time high and is non-positive. Falling days use r_log < -0.02.
    """
    levels = _log_levels(log_index)
    returns = np.diff(levels)
    years = len(returns) / DAYS
    daily_variance = np.var(returns, ddof=1) if len(returns) > 1 else np.nan
    bears = drawdown_episodes(levels)
    intervals = np.diff([event.breach for event in bears]) / DAYS
    metrics = {
        "annual_log_drift": float(returns.mean() * DAYS),
        "annual_simple_drift": float(np.expm1(returns).mean() * DAYS),
        "annual_volatility": float(np.sqrt(daily_variance * DAYS)),
        "excess_kurtosis_daily": _shape(returns, 4),
        "falls_2pct_per_year": float(np.count_nonzero(returns < -0.02) / years),
        "max_log_drawdown": float(np.min(levels - np.maximum.accumulate(levels))),
        "bears_20_per_century": 100 * len(bears) / years,
        "bears_30_per_century": 100 * len(drawdown_episodes(levels, 0.30)) / years,
        "bear_mean_depth": float(np.mean([event.depth for event in bears])) if bears else np.nan,
        "bear_interval_years": float(np.mean(intervals)) if len(intervals) else np.nan,
        "bear_interval_cv": float(np.std(intervals, ddof=1) / np.mean(intervals))
        if len(intervals) > 1 else np.nan,
    }
    for horizon in (21, 63, 126, 252):
        rolling = levels[horizon:] - levels[:-horizon]
        metrics[f"vr_{horizon}"] = (
            float(np.var(rolling, ddof=1) / (horizon * daily_variance))
            if len(rolling) > 1 and daily_variance > 0 else np.nan
        )
    for horizon in (1, 5, 21, 63):
        metrics[f"skew_{horizon}"] = _shape(np.diff(levels[::horizon]), 3)
    return metrics


def _states_array(states):
    levels = np.asarray(states, dtype=float)
    if levels.ndim != 2 or levels.shape[1] != 3 or not np.isfinite(levels).all():
        raise ValueError("states must be a finite array with columns x, y, z")
    return levels


def cross_section_moments(member_log_returns, member_ids, log_index):
    """Table 5 cross-name readings on firms that stay for the full window.

    Both panels are (T+1, slots), as returned by simulate_bbm(keep_panel=True).
    Use a complete fixed set of constituent histories rather than splicing an
    entrant's observations to its predecessor's. Beta is computed relative to
    the daily point-in-time index. The 21-day non-overlapping monthly blocks
    use the sample SD of daily log returns. Take differences in log realized
    volatility before standardizing columns and computing their correlation
    and first principal component. Calendar-month boundaries in the original
    historical panel can differ. The paper does not give an exact survivorship
    convention for each Table 5 cross-sectional row; this choice is explicit.
    """
    levels = _log_levels(log_index)
    returns = np.asarray(member_log_returns, dtype=float)
    ids = np.asarray(member_ids)
    if returns.ndim != 2 or returns.shape != ids.shape or len(returns) != len(levels):
        raise ValueError("return and ID panels must align with each other and log_index")
    metrics = dict.fromkeys(
        ["pair_corr_mean", "pair_corr_sd", "market_beta_sd", "name_vol_mean",
         "name_vol_sd", "vol_change_pc1", "vol_change_corr"], np.nan
    )
    eligible = np.all(ids == ids[0], axis=0) & np.isfinite(returns[1:]).all(axis=0)
    sample = returns[1:, eligible]
    metrics["eligible_survivors"] = sample.shape[1]
    if not sample.shape[1] or len(sample) < 2:
        return metrics

    sd = np.std(sample, axis=0, ddof=1)
    metrics["name_vol_mean"] = float(sd.mean() * np.sqrt(DAYS))
    if sample.shape[1] > 1:
        metrics["name_vol_sd"] = float(sd.std(ddof=1) * np.sqrt(DAYS))
    market = np.diff(levels)
    market -= market.mean()
    if market @ market > 0 and sample.shape[1] > 1:
        betas = (market @ sample) / (market @ market)
        metrics["market_beta_sd"] = float(betas.std(ddof=1))

    def pair_values(corr):
        return corr[np.triu_indices_from(corr, k=1)]

    varying = sample[:, sd > 0]
    if varying.shape[1] >= 2:
        pairs = pair_values(np.corrcoef(varying, rowvar=False))
        metrics["pair_corr_mean"] = float(pairs.mean())
        if len(pairs) > 1:
            metrics["pair_corr_sd"] = float(pairs.std(ddof=1))
    months = len(sample) // 21
    if months >= 4 and varying.shape[1] >= 2:
        blocks = varying[:months * 21].reshape(months, 21, -1)
        vol = blocks.std(axis=1, ddof=1)
        changes = np.diff(np.log(vol[:, np.all(vol > 0, axis=0)]), axis=0)
        changes = changes[:, changes.std(axis=0) > 0]
        if changes.shape[1] >= 2:
            standardized = (changes - changes.mean(axis=0)) / changes.std(axis=0, ddof=1)
            scale = len(standardized) - 1
            # Mean off-diagonal correlation and leading eigenvalue without a large
            # name-by-name eigendecomposition on windows with few monthly samples.
            n = standardized.shape[1]
            sum_columns = standardized.sum(axis=1)
            metrics["vol_change_corr"] = float(
                ((sum_columns @ sum_columns) / scale - n) / (n * (n - 1))
            )
            gram = (standardized @ standardized.T if len(standardized) < n
                    else standardized.T @ standardized)
            metrics["vol_change_pc1"] = float(np.linalg.eigvalsh(gram)[-1] / (scale * n))
    return metrics


def ou_entropy(states, replacement_days=None, days_per_year=DAYS):
    """Estimate Eq. (54), in nats/year, from the daily 3-coordinate path.

    Optional replacement_days has one flag per level (flag t excludes the
    increment ending at t), or one flag per increment. C and A use the selected
    starting levels. D uses actual increments, not regression residuals. Never
    bridge across removed observations. A singular fit has no defined rate.
    """
    levels = _states_array(states)
    starts = levels[:-1]
    increments = np.diff(levels, axis=0)
    if replacement_days is not None:
        excluded = np.asarray(replacement_days, dtype=bool)
        if excluded.shape == (len(levels),):
            excluded = excluded[1:]
        if excluded.shape != (len(increments),):
            raise ValueError("replacement_days must align with levels or increments")
        starts = starts[~excluded]
        increments = increments[~excluded]
    if len(starts) <= 3:
        return np.nan
    scale = starts.std(axis=0)
    if np.any(scale == 0):
        return np.nan
    x = (starts - starts.mean(axis=0)) / scale
    dx = (increments - increments.mean(axis=0)) / scale
    denominator = len(starts) - 1
    covariance = (x.T @ x) / denominator
    diffusion = (dx.T @ dx) * (0.5 * days_per_year / denominator)
    # From the centered increment regression: A*C = -Cov(dx,x)/dt.
    ac = -(dx.T @ x) * (days_per_year / denominator)
    current = (ac - ac.T) / 2
    try:
        left = np.linalg.solve(diffusion, current)
        return float(np.trace(np.linalg.solve(covariance, current.T @ left)))
    except np.linalg.LinAlgError:
        return np.nan


def lag_asymmetry(states, lag=1):
    """Compute the relative asymmetry of Table 30 and signed pairwise asymmetries.

    Cij(lag) has variable i at t and variable j at t+lag. The relative value is
    ||C-C.T||_F / (2||C||_F) in the original coordinates. The pairwise matrix
    divides (C-C.T)/2 by the product of the two unconditional level SDs.
    """
    levels = _states_array(states)
    if not isinstance(lag, (int, np.integer)) or not 0 < lag < len(levels) - 1:
        raise ValueError("lag must leave at least two pairs")
    centered = levels - levels.mean(axis=0)
    covariance = centered[:-lag].T @ centered[lag:] / (len(levels) - lag)
    antisym = (covariance - covariance.T) / 2
    norm = np.linalg.norm(covariance)
    sd = centered.std(axis=0)
    scale = np.outer(sd, sd)
    signed = np.divide(antisym, scale, out=np.full_like(antisym, np.nan), where=scale > 0)
    return float(np.linalg.norm(antisym) / norm) if norm else np.nan, signed
