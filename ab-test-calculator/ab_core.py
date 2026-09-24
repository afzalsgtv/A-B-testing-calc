from dataclasses import dataclass

import numpy as np
from scipy import stats


def _z_crit(alpha: float, two_sided: bool = True) -> float:
    return stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha)


def sample_size_proportions(
    p_base: float,
    mde_abs: float,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> int:
    if mde_abs <= 0:
        raise ValueError("MDE должен быть больше нуля")
    p1, p2 = p_base, p_base + mde_abs
    if not 0 < p1 < 1 or not 0 < p2 < 1:
        raise ValueError("Конверсия и конверсия + MDE должны лежать в (0, 1)")

    p_bar = (p1 + p2) / 2
    numerator = (
        _z_crit(alpha, two_sided) * np.sqrt(2 * p_bar * (1 - p_bar))
        + stats.norm.ppf(power) * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))
    ) ** 2
    return int(np.ceil(numerator / mde_abs**2))


def mde_from_sample_size(
    p_base: float,
    n_per_group: int,
    alpha: float = 0.05,
    power: float = 0.80,
    two_sided: bool = True,
) -> float:
    lo, hi = 1e-6, max(0.99 - p_base, 1e-5)
    if sample_size_proportions(p_base, hi, alpha, power, two_sided) > n_per_group:
        return float("nan")
    for _ in range(80):
        mid = (lo + hi) / 2
        if sample_size_proportions(p_base, mid, alpha, power, two_sided) > n_per_group:
            lo = mid
        else:
            hi = mid
    return hi


def test_duration_days(n_per_group: int, daily_users: int, share_in_test: float = 1.0) -> float:
    per_day = daily_users * share_in_test
    return 2 * n_per_group / per_day if per_day > 0 else float("nan")


@dataclass
class ProportionResult:
    p_a: float
    p_b: float
    diff_abs: float
    diff_ci: tuple[float, float]
    lift_rel: float
    lift_ci: tuple[float, float]
    z_stat: float
    p_value: float
    significant: bool
    alpha: float
    mde_achievable: float


def analyze_proportions(
    n_a: int,
    conv_a: int,
    n_b: int,
    conv_b: int,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> ProportionResult:
    if conv_a > n_a or conv_b > n_b:
        raise ValueError("Конверсий не может быть больше, чем наблюдений")
    if min(n_a, n_b) == 0:
        raise ValueError("Размер группы должен быть больше нуля")

    p_a, p_b = conv_a / n_a, conv_b / n_b
    diff = p_b - p_a

    p_pool = (conv_a + conv_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool if se_pool > 0 else 0.0
    p_value = 2 * stats.norm.sf(abs(z)) if two_sided else stats.norm.sf(z)

    z_crit = _z_crit(alpha, two_sided)
    se_unpool = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    diff_ci = (diff - z_crit * se_unpool, diff + z_crit * se_unpool)

    if p_a > 0 and p_b > 0:
        lift = diff / p_a
        se_log_rr = np.sqrt((1 - p_a) / (n_a * p_a) + (1 - p_b) / (n_b * p_b))
        log_rr = np.log(p_b / p_a)
        lift_ci = (
            np.exp(log_rr - z_crit * se_log_rr) - 1,
            np.exp(log_rr + z_crit * se_log_rr) - 1,
        )
    else:
        lift, lift_ci = float("nan"), (float("nan"), float("nan"))

    return ProportionResult(
        p_a=p_a,
        p_b=p_b,
        diff_abs=diff,
        diff_ci=diff_ci,
        lift_rel=lift,
        lift_ci=lift_ci,
        z_stat=z,
        p_value=p_value,
        significant=p_value < alpha,
        alpha=alpha,
        mde_achievable=mde_from_sample_size(p_a, min(n_a, n_b), alpha, 0.80, two_sided),
    )


@dataclass
class ContinuousResult:
    mean_a: float
    mean_b: float
    diff: float
    diff_ci: tuple[float, float]
    lift_rel: float
    t_stat: float
    p_value: float
    df: float
    significant: bool
    alpha: float


def analyze_continuous(values_a, values_b, alpha: float = 0.05) -> ContinuousResult:
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        raise ValueError("В каждой группе нужно минимум 2 наблюдения")

    n_a, n_b = len(a), len(b)
    va, vb = a.var(ddof=1) / n_a, b.var(ddof=1) / n_b
    se = np.sqrt(va + vb)
    diff = b.mean() - a.mean()
    t_stat = diff / se if se > 0 else 0.0
    df = (va + vb) ** 2 / (va**2 / (n_a - 1) + vb**2 / (n_b - 1))

    p_value = 2 * stats.t.sf(abs(t_stat), df)
    t_crit = stats.t.ppf(1 - alpha / 2, df)

    return ContinuousResult(
        mean_a=a.mean(),
        mean_b=b.mean(),
        diff=diff,
        diff_ci=(diff - t_crit * se, diff + t_crit * se),
        lift_rel=diff / a.mean() if a.mean() != 0 else float("nan"),
        t_stat=t_stat,
        p_value=p_value,
        df=df,
        significant=p_value < alpha,
        alpha=alpha,
    )


def peeking_simulation(
    n_per_group: int,
    n_looks: int,
    p_base: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)

    checkpoints = np.linspace(n_per_group / n_looks, n_per_group, n_looks)
    checkpoints = np.maximum(np.round(checkpoints).astype(int), 1)
    increments = np.diff(np.concatenate([[0], checkpoints]))

    cum_a = np.cumsum(rng.binomial(increments, p_base, size=(n_sims, n_looks)), axis=1)
    cum_b = np.cumsum(rng.binomial(increments, p_base, size=(n_sims, n_looks)), axis=1)

    n = checkpoints[None, :]
    p_pool = (cum_a + cum_b) / (2 * n)
    se = np.sqrt(np.clip(p_pool * (1 - p_pool) * 2 / n, 1e-12, None))
    z = (cum_b - cum_a) / n / se
    p_values = 2 * stats.norm.sf(np.abs(z))

    return {
        "naive": float((p_values < alpha).any(axis=1).mean()),
        "bonferroni": float((p_values < alpha / n_looks).any(axis=1).mean()),
        "final_only": float((p_values[:, -1] < alpha).mean()),
        "n_looks": n_looks,
        "alpha": alpha,
        "n_sims": n_sims,
    }


def peeking_curve(
    n_per_group: int,
    max_looks: int = 10,
    p_base: float = 0.10,
    alpha: float = 0.05,
    n_sims: int = 2000,
    seed: int = 42,
) -> list[dict]:
    return [
        peeking_simulation(n_per_group, k, p_base, alpha, n_sims, seed + k)
        for k in range(1, max_looks + 1)
    ]
