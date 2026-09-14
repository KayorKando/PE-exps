"""Privacy accounting (spec section 4).

sigma is *derived* from (eps, delta, T), never hand-set.

Mechanism.  Each round every private point casts exactly one vote for its
nearest synthetic point, so under add/remove-one neighbouring the count
histogram changes by 1 in a single bin: L2 sensitivity 1 per round.  We
release T such histograms, each with i.i.d. N(0, sigma^2) noise per bin.

Two accountants are provided and both are reported:

* ``rdp``      -- RDP of the Gaussian mechanism, alpha/(2 sigma^2) per round,
                  composed additively over T rounds, converted to (eps, delta)
                  with the tightened conversion of Canonne-Kamath-Steinke.
                  This is what Aug-PE uses; it is the default.
* ``analytic`` -- the T releases jointly form one Gaussian mechanism with L2
                  sensitivity sqrt(T), so the exact analytic Gaussian
                  mechanism calibration (Balle & Wang, ICML 2018) applies.
                  Slightly tighter; reported as a cross-check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import numpy as np
from scipy import optimize
from scipy.stats import norm

# Standard alpha grid (same shape as the opacus / dp_accounting default).
DEFAULT_ALPHAS = (
    [1 + x / 10.0 for x in range(1, 100)]
    + list(range(11, 64))
    + [128, 256, 512, 1024]
)


def rdp_gaussian(sigma: float, steps: int, alphas=DEFAULT_ALPHAS) -> np.ndarray:
    """RDP epsilon at each order for `steps` compositions, sensitivity 1."""
    a = np.asarray(alphas, dtype=float)
    return steps * a / (2.0 * sigma**2)


def rdp_to_dp(rdp: np.ndarray, delta: float, alphas=DEFAULT_ALPHAS):
    """Convert RDP curve to (eps, delta)-DP.  Returns (eps, best_alpha).

    Uses the improved conversion (Canonne, Kamath & Steinke 2020), which is
    what opacus and dp_accounting implement:

        eps = rdp + log((alpha-1)/alpha) - (log(delta) + log(alpha))/(alpha-1)
    """
    a = np.asarray(alphas, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = (
            rdp
            + np.log1p(-1.0 / a)
            - (np.log(delta) + np.log(a)) / (a - 1.0)
        )
    eps = np.where(np.isfinite(eps), eps, np.inf)
    eps = np.maximum(eps, 0.0)
    i = int(np.argmin(eps))
    return float(eps[i]), float(a[i])


def analytic_gaussian_sigma(eps: float, delta: float, sensitivity: float = 1.0) -> float:
    """Exact analytic Gaussian mechanism calibration (Balle & Wang 2018, Alg. 1)."""

    def b_plus(v):
        return norm.cdf(math.sqrt(eps * v)) - math.exp(eps) * norm.cdf(
            -math.sqrt(eps * (v + 2.0))
        )

    def b_minus(v):
        return norm.cdf(-math.sqrt(eps * v)) - math.exp(eps) * norm.cdf(
            -math.sqrt(eps * (v + 2.0))
        )

    delta_0 = b_plus(0.0)  # both branches agree at v = 0
    if delta >= delta_0:
        f, increasing, alpha_sign = b_plus, True, -1.0
    else:
        f, increasing, alpha_sign = b_minus, False, +1.0

    hi = 1.0
    while (f(hi) < delta) if increasing else (f(hi) > delta):
        hi *= 2.0
        if hi > 1e12:
            raise RuntimeError("analytic Gaussian: failed to bracket")
    v_star = optimize.brentq(lambda v: f(v) - delta, 0.0, hi, xtol=1e-13, rtol=1e-14)

    alpha = math.sqrt(1.0 + v_star / 2.0) + alpha_sign * math.sqrt(v_star / 2.0)
    return alpha * sensitivity / math.sqrt(2.0 * eps)


@dataclass(frozen=True)
class PrivacySpend:
    eps: float
    delta: float
    T: int
    sigma: float
    accountant: str
    rdp_order: float
    sigma_rdp: float
    sigma_analytic: float

    def as_dict(self):
        return asdict(self)


def derive_sigma(
    eps: float, delta: float, T: int, accountant: str = "rdp"
) -> PrivacySpend:
    """Solve for the per-count noise sigma achieving (eps, delta) over T rounds."""
    if accountant not in ("rdp", "analytic"):
        raise ValueError(f"unknown accountant {accountant!r}")

    def eps_of_sigma(sigma):
        e, _ = rdp_to_dp(rdp_gaussian(sigma, T), delta)
        return e

    lo, hi = 1e-3, 1.0
    while eps_of_sigma(hi) > eps:
        hi *= 2.0
        if hi > 1e6:
            raise RuntimeError("derive_sigma: failed to bracket")
    sigma_rdp = optimize.brentq(
        lambda s: eps_of_sigma(s) - eps, lo, hi, xtol=1e-10, rtol=1e-12
    )
    _, order = rdp_to_dp(rdp_gaussian(sigma_rdp, T), delta)

    # T rounds of sensitivity-1 Gaussian == one release of sensitivity sqrt(T).
    sigma_analytic = analytic_gaussian_sigma(eps, delta, sensitivity=math.sqrt(T))

    sigma = sigma_rdp if accountant == "rdp" else sigma_analytic
    return PrivacySpend(
        eps=eps,
        delta=delta,
        T=T,
        sigma=float(sigma),
        accountant=accountant,
        rdp_order=order,
        sigma_rdp=float(sigma_rdp),
        sigma_analytic=float(sigma_analytic),
    )


if __name__ == "__main__":
    for e in (0.5, 1.0, 4.0):
        sp = derive_sigma(e, 1.0 / 2000, 10)
        print(
            f"eps={e:<4} delta=1/2000 T=10 -> sigma_rdp={sp.sigma_rdp:8.4f} "
            f"(alpha*={sp.rdp_order:g})  sigma_analytic={sp.sigma_analytic:8.4f}"
        )
