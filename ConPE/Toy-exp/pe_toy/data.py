"""Data generation (spec section 3).

Config A: known Gaussian mixture -> ground-truth log p and grad log p are
available to the *metrics* module and to the `directed_truth` diagnostic arm
only.  Config B: sentence-encoder embeddings of a public corpus, PCA'd to D.
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import time
from dataclasses import dataclass, field

import numpy as np
from scipy.special import logsumexp

CACHE_DIR = pathlib.Path(__file__).resolve().parent.parent / ".cache"


# --------------------------------------------------------------------------
# Config A: synthetic mixture
# --------------------------------------------------------------------------


@dataclass
class GaussianMixture:
    """A mixture with analytic log-density and score, for diagnostics only."""

    weights: np.ndarray  # (M,)
    mus: np.ndarray  # (M, D)
    covs: np.ndarray  # (M, D, D)
    chols: np.ndarray = field(repr=False, default=None)  # (M, D, D)
    precs: np.ndarray = field(repr=False, default=None)  # (M, D, D)
    logdets: np.ndarray = field(repr=False, default=None)  # (M,)

    def __post_init__(self):
        if self.chols is None:
            self.chols = np.linalg.cholesky(self.covs)
        if self.precs is None:
            self.precs = np.linalg.inv(self.covs)
        if self.logdets is None:
            self.logdets = 2.0 * np.log(
                np.diagonal(self.chols, axis1=1, axis2=2)
            ).sum(axis=1)

    @property
    def D(self):
        return self.mus.shape[1]

    @property
    def M(self):
        return self.mus.shape[0]

    def sample(self, n, rng, modes=None, weights=None):
        w = self.weights if weights is None else weights
        idx = np.arange(self.M) if modes is None else np.asarray(modes)
        w = w[idx] / w[idx].sum()
        which = rng.choice(idx, size=n, p=w)
        eps = rng.standard_normal((n, self.D))
        out = np.empty((n, self.D))
        for m in np.unique(which):
            sel = which == m
            out[sel] = self.mus[m] + eps[sel] @ self.chols[m].T
        return out

    def _log_components(self, Z):
        """(n, M) log w_m + log N(z; mu_m, Sigma_m)."""
        n = Z.shape[0]
        out = np.empty((n, self.M))
        const = self.D * np.log(2.0 * np.pi)
        for m in range(self.M):
            d = Z - self.mus[m]
            maha = np.einsum("ni,ij,nj->n", d, self.precs[m], d)
            out[:, m] = (
                np.log(self.weights[m]) - 0.5 * (maha + self.logdets[m] + const)
            )
        return out

    def log_prob(self, Z):
        return logsumexp(self._log_components(Z), axis=1)

    def score(self, Z):
        """grad_z log p(z), shape (n, D)."""
        lc = self._log_components(Z)
        r = np.exp(lc - logsumexp(lc, axis=1, keepdims=True))  # (n, M)
        g = np.zeros_like(Z)
        for m in range(self.M):
            d = Z - self.mus[m]
            g -= r[:, m, None] * (d @ self.precs[m].T)
        return g


def _place_centers(M, D, target, rng, iters=600):
    """M points spread out so their median pairwise distance is `target`."""
    C = rng.standard_normal((M, D))
    for _ in range(iters):
        diff = C[:, None, :] - C[None, :, :]
        d = np.linalg.norm(diff, axis=-1)
        np.fill_diagonal(d, np.inf)
        force = (diff / (d[..., None] ** 3 + 1e-12)).sum(axis=1)
        C = C + 0.02 * force / (np.abs(force).max() + 1e-12) * np.median(d[np.isfinite(d)])
        C -= C.mean(axis=0)
    d = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=-1)
    med = np.median(d[~np.eye(M, dtype=bool)])
    return C * (target / med)


def make_mixture(D, M=8, sep_sd=2.5, eig_lo=0.3, eig_hi=3.0, seed=0):
    rng = np.random.default_rng(1000 + seed)
    w = 1.0 / np.arange(1, M + 1)
    w = w / w.sum()

    eigs = np.exp(rng.uniform(np.log(eig_lo), np.log(eig_hi), size=(M, D)))
    covs = np.empty((M, D, D))
    for m in range(M):
        Q, _ = np.linalg.qr(rng.standard_normal((D, D)))
        covs[m] = (Q * eigs[m]) @ Q.T
        covs[m] = 0.5 * (covs[m] + covs[m].T)

    # "adjacent modes overlap at 2-3 s.d.": separate centres by sep_sd times
    # the typical marginal s.d. of the components.
    typical_sd = float(np.sqrt(eigs.mean()))
    mus = _place_centers(M, D, sep_sd * typical_sd, rng)
    return GaussianMixture(weights=w, mus=mus, covs=covs)


# --------------------------------------------------------------------------
# Config B: real-encoder embeddings
# --------------------------------------------------------------------------

_CORPUS_SPECS = {
    "yelp": dict(
        dataset="fancyzhx/yelp_polarity", config="plain_text", split="train", col="text"
    ),
    "pubmed": dict(
        dataset="ccdv/pubmed-summarization", config="section", split="train", col="abstract"
    ),
}


def fetch_corpus(name="yelp", n_sentences=8000, cache=True):
    """Pull sentences from the HF datasets-server (no `datasets` dep needed)."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"corpus_{name}_{n_sentences}.json"
    if cache and path.exists():
        return json.loads(path.read_text())

    import urllib.parse
    import urllib.request

    spec = _CORPUS_SPECS[name]
    out, offset = [], 0
    while len(out) < n_sentences:
        q = urllib.parse.urlencode(
            dict(
                dataset=spec["dataset"],
                config=spec["config"],
                split=spec["split"],
                offset=offset,
                length=100,
            )
        )
        url = f"https://datasets-server.huggingface.co/rows?{q}"
        rows = None
        for attempt in range(6):  # the rows endpoint 502s intermittently
            try:
                with urllib.request.urlopen(url, timeout=90) as r:
                    rows = json.load(r)["rows"]
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2.0 * (attempt + 1))
        if not rows:
            break
        for row in rows:
            txt = " ".join(str(row["row"][spec["col"]]).split())
            # crude sentence split; keep sentences of a usable length
            for s in txt.replace("!", ".").replace("?", ".").split("."):
                s = s.strip()
                if 40 <= len(s) <= 300:
                    out.append(s)
        offset += 100
        if offset > 40000:
            break
    out = out[:n_sentences]
    if cache:
        path.write_text(json.dumps(out))
    return out


def embed_corpus(sentences, model_name="sentence-transformers/all-mpnet-base-v2"):
    key = hashlib.sha1(
        (model_name + str(len(sentences)) + sentences[0][:64]).encode()
    ).hexdigest()[:16]
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"emb_{key}.npy"
    if path.exists():
        return np.load(path)
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    E = model.encode(
        sentences, batch_size=64, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float64)
    np.save(path, E)
    return E


def _configB_z0(corpus, D, N, n_syn, seed, P):
    """Z_0 for config B: sampled from the heavier of a 2-component GMM fit to P.

    Cached on disk.  The fit is by far the most expensive thing in the whole
    harness -- 62 s of a 68 s D=768 run, 28 EM iterations over a 20000x768
    matrix with two full covariances -- and it depends only on
    (corpus, D, N, n_syn, seed), not on the arm, eps, theta, k or alpha.  Stage 5
    at D=768 is 60 runs over just 5 distinct datasets, so without this cache 55
    of the 60 fits are recomputing a value they already had.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"z0_{corpus}_D{D}_N{N}_n{n_syn}_s{seed}.npy"
    if path.exists():
        return np.load(path)
    from sklearn.mixture import GaussianMixture as SkGMM

    gm = SkGMM(n_components=2, covariance_type="full", random_state=seed).fit(P)
    heavier = int(np.argmax(gm.weights_))
    cov = gm.covariances_[heavier]
    L = np.linalg.cholesky(cov + 1e-6 * np.eye(D) * np.trace(cov) / D)
    Z0 = gm.means_[heavier] + np.random.default_rng(seed).standard_normal(
        (n_syn, D)
    ) @ L.T
    np.save(path, Z0)
    return Z0


def pca_pool(corpus, D, n_sentences=24000):
    """Corpus embeddings projected to D dims, cached as a small .npy."""
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"pca_{corpus}_{n_sentences}_{D}.npy"
    if path.exists():
        return np.load(path)
    from sklearn.decomposition import PCA

    E = embed_corpus(fetch_corpus(corpus, n_sentences=n_sentences))
    X = PCA(n_components=D, random_state=0).fit_transform(E)
    X = X / X.std(axis=0).mean()  # unit-ish scale, keeps the anisotropy
    np.save(path, X)
    return X


# --------------------------------------------------------------------------
# Datasets handed to the pipeline
# --------------------------------------------------------------------------


@dataclass
class Dataset:
    P: np.ndarray  # (N, D) private
    Z0: np.ndarray  # (n_syn, D) initial synthetic
    reference: np.ndarray  # (n_ref, D) held-out sample for metrics
    config: str
    D: int
    mixture: GaussianMixture | None = None  # config A only, = p
    q0: GaussianMixture | None = None  # config A only, = the law of Z_0
    reference_is_private: bool = False

    @property
    def N(self):
        return self.P.shape[0]


def make_dataset(
    config="A",
    D=128,
    N=20000,
    n_syn=500,
    n_ref=5000,
    seed=0,
    q0_noise_sd=1.5,
    corpus="yelp",
):
    if config == "A":
        # Mixture geometry is fixed across seeds (it is the "task"); the
        # sampling of P, Z0 and the reference varies with the seed.
        mix = make_mixture(D, seed=0)
        rng = np.random.default_rng(seed)
        P = mix.sample(N, rng)
        ref = mix.sample(n_ref, rng)
        # q0: heads-only, over-dispersed -- an LLM prior that misses the tail.
        heavy = np.argsort(mix.weights)[::-1][:2]
        Z0 = mix.sample(n_syn, rng, modes=heavy)
        Z0 = Z0 + q0_noise_sd * rng.standard_normal(Z0.shape)
        # The law of Z_0 is known in closed form -- heaviest two components
        # convolved with isotropic noise -- so grad log q_0 needs no KDE.
        # This matters for Q0: run #1's metric 5 was contaminated by KDE
        # jitter in the reference vector (spec v0.2 D2).
        w0 = mix.weights[heavy] / mix.weights[heavy].sum()
        q0 = GaussianMixture(
            weights=w0,
            mus=mix.mus[heavy],
            covs=mix.covs[heavy] + q0_noise_sd**2 * np.eye(D)[None],
        )
        return Dataset(P=P, Z0=Z0, reference=ref, config="A", D=D,
                       mixture=mix, q0=q0)

    if config == "B":
        # The PCA projection is cached on disk: refitting a 24000x768 PCA in
        # every worker process is what OOM-killed the first stage-1 attempt.
        X = pca_pool(corpus, D, n_sentences=max(24000, N + n_ref))
        rng = np.random.default_rng(seed)
        X = X[rng.permutation(len(X))]
        P = X[:N]
        # Spec 7: for config B the reference is the private set itself.  With
        # N = 20,000 that makes every metric call an O(N^2) k-NN problem
        # (2.3 GB and ~90 s per run at D=768), so the reference is a random
        # n_ref-subset of P rather than all of it -- same distribution, and the
        # same reference size as config A, which keeps the two comparable.
        ref = P[np.random.default_rng(1234 + seed).choice(N, min(n_ref, N),
                                                          replace=False)]
        Z0 = _configB_z0(corpus, D, N, n_syn, seed, P)
        return Dataset(
            P=P,
            Z0=Z0,
            reference=ref,
            config="B",
            D=D,
            reference_is_private=True,
        )

    raise ValueError(f"unknown config {config!r}")
