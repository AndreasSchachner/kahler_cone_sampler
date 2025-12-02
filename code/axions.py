# ===================================================================================
# Copyright 2025 Elli Heyes, Mudit Jain, David J. E. Marsh, Keir K. Rogers, 
# Andreas Schachner, and Elijah Sheridan
#
#   This script validates the results from complex structure moduli stabilization in 
#   the various de Sitter and anti-de Sitter vacua obtained in ArXiv:2406.13751.
#
#   In the event of bugs or other issues, please reach out via as3475@cornell.edu
#   or a.schachner@lmu.de.
#
# ===================================================================================
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------------


from __future__ import annotations

import numpy as np
from typing import Union


from scipy.linalg import lu_factor, lu_solve, solve_triangular, cho_factor, cho_solve
from mpmath import mp

import numpy as np
from scipy.linalg import cho_factor, cho_solve, pinvh






# ---- Stable weighted Gram (all-mp) ----
def stable_weighted_gram_mp(N, logw):
    """
    **Description:**
    Compute the weighted Gram matrix H = N W N^T with weights wj = exp(logw[j])
    in a numerically stable way, returning also the scale s = max_j logw[j].
    
    Args:
        N: mp.matrix (r,J)
        logw: iterable of length J of mp.mpf (log weights)
        
    Returns:
        H: mp.matrix (r,r) weighted Gram matrix
        s: mp.mpf scale = max_j logw[j]
    """
    r, J = N.rows, N.cols
    logw = [mp.mpf(x) for x in logw]
    s = max(logw) if J > 0 else mp.ninf
    H = mp.matrix(r, r)
    for j in range(J):
        # weight scaled so that max(log w) -> 0
        wj = mp.e**(logw[j] - s)   # in (0, 1]
        # rank-1 update: wj * (N[:,j] * N[:,j]^T)
        for a in range(r):
            Na = N[a, j]
            if Na == 0: 
                continue
            for b in range(r):
                Nb = N[b, j]
                if Nb != 0:
                    H[a, b] += wj * Na * Nb
    return H, s

# ---- Cholesky (lower) with tiny jitter if needed ----
def cholesky_lower_mp(K, max_tries=5):
    """
    **Description:**
    Compute the lower-triangular Cholesky factor L of a symmetric matrix K,
    adding tiny jitter on the diagonal if needed for positive definiteness.
    
    Args:
        K: symmetric mp.matrix.
        max_tries: int (maximum number of jitter escalations).
        
    Returns:
        L: lower-triangular mp.matrix such that K ≈ L L^T.
        
    """
    n = K.rows
    # Make symmetric
    Ksym = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            Ksym[i,j] = (K[i,j] + K[j,i]) / 2

    # Try vanilla Cholesky; if it fails, add jitter progressively
    # Note: mpmath provides mp.cholesky for Hermitian positive definite matrices
    jitter = mp.mpf('0')
    tr = sum(Ksym[i,i] for i in range(n))
    scale = tr / n if n > 0 else mp.one

    for t in range(max_tries):
        try:
            if jitter > 0:
                A = mp.matrix(n, n)
                for i in range(n):
                    for j in range(n):
                        A[i,j] = Ksym[i,j]
                    A[i,i] += jitter
            else:
                A = Ksym
            L = mp.cholesky(A)  # lower-triangular
            return L
        except Exception:
            # increase jitter geometrically
            if scale == 0:
                scale = mp.one
            if jitter == 0:
                jitter = mp.eps * scale
            else:
                jitter *= 10
    # last resort: raise
    raise ValueError("Cholesky failed; matrix may be too indefinite even with jitter.")

# ---- Triangular solves ----
def forward_substitute(L, B):
    """
    **Description:**
    Solve L X = B for X, L lower-triangular.
    
    Args:
        L: Lower-triangular mp.matrix.
        B: mp.matrix.
        
    Returns:
        X: mp.matrix solution of L X = B.
    """
    n = L.rows
    m = B.cols
    X = mp.matrix(n, m)
    for j in range(m):
        for i in range(n):
            s = B[i, j]
            for k in range(i):
                s -= L[i, k] * X[k, j]
            X[i, j] = s / L[i, i]
    return X

def backward_substitute(U, B):
    """
    **Description:**
    Solve U X = B for X, U upper-triangular.
    
    Args:
        U: Upper-triangular mp.matrix.
        B: mp.matrix.
        
    Returns:
        X: mp.matrix solution of U X = B.
    """
    n = U.rows
    m = B.cols
    X = mp.matrix(n, m)
    for j in range(m):
        for i in reversed(range(n)):
            s = B[i, j]
            for k in range(i+1, n):
                s -= U[i, k] * X[k, j]
            X[i, j] = s / U[i, i]
    return X



def diagonalize_lagrangian_mp(K, M2, clip_neg=True):
    """
    **Description:**
    Diagonalize the generalized eigenvalue problem
      M2 v = λ K v
    for symmetric positive definite K and symmetric M2,
    returning the masses and eigenvectors.
    
    Args:
      K  : mp.matrix (r,r)   symmetric positive definite kinetic matrix
      M2 : mp.matrix (r,r)   symmetric mass-squared matrix
      clip_neg : bool        whether to clip tiny negative eigenvalues to zero
    
    Returns:
      m : mp.matrix (r,1)   masses >=0
      V : mp.matrix (r,r)   eigenvectors in original basis (V^T K V = I)
      U : mp.matrix (r,r)   canonical-basis eigenvectors (U^T U = I)
      L : mp.matrix (r,r)   lower Cholesky of K (K ≈ L L^T)
    """
    # 0) Symmetrize inputs
    K  = symmetrize(K)
    M2 = symmetrize(M2)

    # 1) Cholesky of K (add tiny jitter inside cholesky_lower_mp if needed)
    L = cholesky_lower_mp(K)
    Lt = L.T

    # 2) Canonical mass matrix Mtilde = L^{-1} M2 L^{-T}
    B       = forward_substitute(L, M2)      # left solve
    Mtilde  = forward_substitute(L, B.T).T   # right solve via transpose
    Mtilde  = symmetrize(Mtilde)

    # 3) Eigendecomposition in canonical basis
    w_list, U = mp.eig(Mtilde)               # columns of U are eigenvectors
    # real-clean eigenvalues
    w = [mp.re(v) if abs(mp.im(v)) <= mp.eps**0.5 else v for v in w_list]

    # 4) Sort ascending
    idx = sorted(range(len(w)), key=lambda i: w[i])
    w_sorted = [w[i] for i in idx]
    U_sorted = mp.matrix(U.rows, U.cols)
    for k,i in enumerate(idx):
        for r in range(U.rows):
            U_sorted[r, k] = U[r, i]

    # 5) Clip tiny negative noise and take sqrt
    if clip_neg:
        tiny = mp.mpf('1e-120')
        w_sorted = [val if val > tiny else tiny for val in w_sorted]
    m = mp.matrix(len(w_sorted), 1)
    for i,val in enumerate(w_sorted):
        m[i] = mp.sqrt(val)

    # 6) Orthonormalize U (Euclidean) – Gram–Schmidt
    U_orth = mp.matrix(U_sorted.rows, U_sorted.cols)
    for j in range(U_sorted.cols):
        v = mp.matrix(U_sorted.rows, 1)
        for r in range(U_sorted.rows): v[r] = U_sorted[r, j]
        for k in range(j):
            uk = U_orth[:, k]
            coeff = (uk.T * v)[0]
            v -= uk * coeff
        norm = mp.sqrt((v.T * v)[0])
        if norm != 0: v /= norm
        for r in range(U_sorted.rows): U_orth[r, j] = v[r]
    U = U_orth

    # 7) Map back to original basis: V = L^{-T} U  (solve Lt V = U)
    V = backward_substitute(Lt, U)

    return m, V, U, L





# ---- Convenience: log-masses with scaling from stable_weighted_gram_mp ---- #
def log10_masses_from_scaled(m, s, eps=mp.mpf('1e-80')):
    """
    **Description:**
    Compute log10 of masses from scaled masses m and scale s.
    
    Args:
        m: mp.matrix (r,1) of masses
        s: mp.mpf scale from stable_weighted_gram_mp
        eps: mp.mpf minimum mass to avoid log(0)
        
    Returns:
        mp.matrix (r,1) of log10 masses.
    """
    r = m.rows
    out = mp.matrix(r, 1)
    for i in range(r):
        mi = m[i]
        if eps and mi < eps:
            mi = eps
        out[i] = mp.log10(mi) + s / (2 * mp.log(10))
    return out


def mp_to_numpy(M):
    """
    **Description:**
    Convert an mp.matrix M to a numpy array of dtype float64.
    
    Args:
        M: mp.matrix
        
    Returns:
        numpy array with dtype float64.
    """
    return np.array(M.tolist(), dtype=np.float64)




def inv_spd_cholesky(K, jitter0=0.0, max_tries=6, return_info=False, compute_logdet=False):
    """
    **Description:**
    Invert a symmetric (near-SPD) matrix using Cholesky + jitter.
    Falls back to Hermitian pseudo-inverse if needed.

    Args:
        K: 2D numpy array (symmetric matrix)
        jitter0: float (initial jitter to add on diagonal)
        max_tries: int (maximum number of jitter escalations)
        return_info: bool (whether to return info dict)
        compute_logdet: bool (whether to compute log-determinant)
        
    Returns:
        Kinv: symmetric numpy array K^{-1}
        (optionally) info: dict with keys 'success', 'jitter_used', 'method', 'logdet'
    """
    K = np.array(K, copy=False)
    K = (K + K.T) * 0.5  # enforce symmetry
    n = K.shape[0]
    I = np.eye(n, dtype=K.dtype)

    jitter = jitter0
    method = 'cholesky'
    info = {'success': False, 'jitter_used': 0.0, 'method': method, 'logdet': None}

    # Robust scale for jitter (handles tiny or badly scaled K)
    scale = max(np.linalg.norm(K, ord=np.inf), np.finfo(K.dtype).tiny)

    for _ in range(max_tries):
        try:
            Kf = K if jitter == 0 else (K + jitter * scale * I)
            c, lower = cho_factor(Kf, lower=True, check_finite=False)
            Kinv = cho_solve((c, lower), I, check_finite=False)

            # Optional logdet from Cholesky: logdet(Kf) = 2 * sum(log(diag(L)))
            if compute_logdet:
                L = np.tril(c) if lower else np.triu(c).T
                # diag(L) positive for SPD; guard tiny values
                diagL = np.diag(L)
                # avoid log(0)
                diagL = np.clip(diagL, np.finfo(K.dtype).tiny, None)
                logdet = 2.0 * np.sum(np.log(diagL))
                info['logdet'] = float(logdet)

            info['success'] = True
            info['jitter_used'] = float(jitter)
            info['method'] = 'cholesky'
            Kinv = (Kinv + Kinv.T) * 0.5
            return (Kinv, info) if return_info else Kinv

        except np.linalg.LinAlgError:
            # escalate jitter tied to dtype precision
            eps = np.finfo(K.dtype).eps
            jitter = eps if jitter == 0 else jitter * 10

    # Fallback: Hermitian pseudo-inverse (handles semi-definite / near-singular)
    Kinv = pinvh(K, check_finite=False)
    Kinv = (Kinv + Kinv.T) * 0.5
    info.update({'success': False, 'jitter_used': float(jitter), 'method': 'pinvh'})
    return (Kinv, info) if return_info else Kinv


def inv_symmetric_eigh(K, rcond=1e-12):
    """
    **Description:**
    Invert a symmetric matrix using its eigendecomposition with eigenvalue clipping.
    
    Args:
        K: 2D numpy array (symmetric matrix)
        rcond: float (relative condition number for eigenvalue clipping)
        
    Returns:
        Symmetric numpy array K^{-1}.
    """
    K = (K + K.T) * 0.5
    w, Q = np.linalg.eigh(K)
    wmax = np.max(np.abs(w))
    thr = rcond * wmax if wmax > 0 else rcond
    w = np.where(w < thr, thr, w)
    Kinv = (Q * (1.0 / w)) @ Q.T
    return (Kinv + Kinv.T) * 0.5


def get_basis_divisor_volumes(Kijk, T):
    """
    **Description:**
    Compute the basis divisor volumes tau_i = 0.5 * k_ijk t^j t^k.
    
    Args:
        Kijk: 3D numpy array of triple intersection numbers.
        T: 1D numpy array of 2-cycle volumes. (h11,) or (m, h11)  -> scalar or (m,)
        
    Returns:
        Basis divisor volumes as float or 1D numpy array. (h11,) or (m, h11)
    """
    T = np.asarray(T, dtype=np.longdouble)
    single = (T.ndim == 1)
    T2 = np.atleast_2d(T)                     # (m, h11)
    tau = 0.5 * np.einsum('ijk,mi,mj->mk', Kijk, T2, T2, optimize=True)  # (m, h11)
    return tau[0] if single else tau

def get_divisor_volumes(Kijk, T, charges):
    """
    **Description:**
    Compute the volumes of divisors given their charges.
    
    Args:
        Kijk: 3D numpy array of triple intersection numbers.
        T: 1D numpy array of 2-cycle volumes. (h11,) or (m, h11)  -> scalar or (m,)
        charges: 2D numpy array of divisor charges. (n_div, h11)
        
    Returns:
        Divisor volumes as float or 1D numpy array. (n_div,) or (m, n_div)
        
    """
    T = np.asarray(T, dtype=np.longdouble)
    single = (T.ndim == 1)
    T2 = np.atleast_2d(T)                     # (m, h11)
    tau = get_basis_divisor_volumes(Kijk, T2)            # (m, h11)
    out = np.asarray(tau, dtype=np.longdouble) @ \
          np.asarray(charges, dtype=np.longdouble).T           # (m, n_div)
    return out[0] if single else out

def get_cy_volume(Kijk, T):
    """
    **Description:**
    Compute the Calabi-Yau volume.
    
    Args:
        Kijk: 3D numpy array of triple intersection numbers.
        T: 1D numpy array of 2-cycle volumes. (h11,) or (m, h11)  -> scalar or (m,)
    
    Returns:
        Calabi-Yau volume as float or 1D numpy array.
    """
    T = np.asarray(T, dtype=np.longdouble)
    single = (T.ndim == 1)
    T2 = np.atleast_2d(T)                     # (m, h11)
    V = (1.0/6.0) * np.einsum('ijk,mi,mj,mk->m', Kijk, T2, T2, T2, optimize=True)
    V = V.astype(np.longdouble)
    return V[0] if single else V

def get_inverse_Kahler_metric(Kijk, tset):
    """
    **Description:**
    Compute the inverse K metric used in axion decay constants.
    
    Args:
        Kijk: 3D numpy array of triple intersection numbers.
        tset: 1D numpy array of 2-cycle volumes.
    
    Returns:
        Symmetric numpy array K_ij^{-1}.
    """
    kijlower = np.einsum('ijk,k->ij', Kijk, tset, optimize=True)  # k_ij
    tau = get_basis_divisor_volumes(Kijk, tset)
    V = get_cy_volume(Kijk, tset)
    # Kmetric = get_Kahler_metric(Kijk, tset)
    # Kmetric_inverse = np.linalg.inv(Kmetric)
    Kmetric_inverse = 2*np.outer(tau,tau) - 2*V*kijlower
    return (Kmetric_inverse + Kmetric_inverse.T)/2


def get_Kahler_metric(Kijk, tset):
    """
    **Description:**
    Compute the K metric used in axion decay constants.
    
    Args:
        Kijk: 3D numpy array of triple intersection numbers.
        tset: 1D numpy array of 2-cycle volumes.
        
    Returns:
        Symmetric numpy array K_ij.
    """
    Kinv = get_inverse_Kahler_metric(Kijk, tset)
    Kmetric = inv_spd_cholesky(Kinv)
    return (Kmetric + Kmetric.T)/2


def get_Kahler_metric_odd(K_eoo, K_eee, tset_even, gs: float = 0.5):
    """ 
    **Description:**
    Compute the K_odd metric used in axion decay constants.
    
    Args:
        K_eoo: 3D numpy array of mixed intersection numbers.
        K_eee: 3D numpy array of even intersection numbers.
        tset_even: 1D numpy array of even 2-cycle volumes.
        gs: string coupling (default 0.5).
        
    Returns:
        Symmetric numpy array K_odd.
    """
    V = get_cy_volume(K_eee, tset_even)
    Kmetric_odd = - 0.5 * gs * np.einsum('ijk,i->jk', K_eoo, tset_even, optimize=True) / V
    return (Kmetric_odd + Kmetric_odd.T)/2


def get_M_matrix(Kijk, tset):
    """ 
    **Description:**
    Compute the M_ij matrix used in axion decay constants.
    
    Args:
        Kijk: 3D numpy array of triple intersection numbers.
        tset: 1D numpy array of 2-cycle volumes.
        
    Returns:
        Symmetric numpy array M_ij.
    """
    V = get_cy_volume(Kijk, tset)
    kijlower = np.einsum('ijk,k->ij', Kijk, tset, optimize=True)  # k_ij
    tau = get_basis_divisor_volumes(Kijk, tset)
    M = (2*np.outer(tau,tau) - V*kijlower)/(2*V**2)
    return (M + M.T)/2


# Extra prefactors (can be modified if the derivation changes)
def logextra(divvols):   # array
    """
    **Description:**
    Extra logarithmic prefactor in axion decay constants.
    
    Args:
        divvols: array-like of divisor volumes (float)
        
    Returns:
        numpy array of log extra factors.
    """
    return np.log(2.0 * (1.0 + 2.0 * 2*np.pi * divvols))

# K_modulus part (dimensionless; Planck units): K = -2 ln Vol - ln(2/gs) + ln P
def K_modulus(CYvol, gs: float = 0.5):
    """
    **Description:**
    Compute the Kähler modulus K = -2 ln Vol - ln(2/gs) + ln P.
    
    Args:
        CYvol: float (Calabi-Yau volume)
        gs: string coupling (default 0.5)
        
    Returns:
        float Kähler modulus K.
    """
    P = (gs**3) / 64.0
    return -2.0 * np.log(CYvol) + 1*(- np.log(2/gs) + np.log(P))




def select_independent_columns_fast(GLSM_sorted, h11, tol=1e-12):
    """
    **Description:**
    Greedily select linearly independent columns of GLSM_sorted in order of
    increasing ~Lambda^4 scale (column order). Stop when no further independent
    columns are found or when h11 are selected.

    Args:
        GLSM_sorted : numpy.ndarray
            2D array of shape (h11, n_cols) with GLSM charge vectors as columns,
            sorted by increasing scale.
        h11 : int
            Number of rows in GLSM_sorted (dimension of the space).
        tol : float
            Tolerance for linear independence (default: 1e-12).

    Returns:
        idx : list[int]
            0-based indices of the selected columns (<= h11 in length).
    """
    n_rows, n_cols = GLSM_sorted.shape
    assert n_rows == h11, "GLSM_sorted must have h11 rows"

    idx = []
    basis = []

    for j in range(n_cols):
        v = GLSM_sorted[:, j].astype(float)

        # subtract projections onto current basis
        for q in basis:
            v = v - np.dot(q, v) * q

        norm_v = np.linalg.norm(v)
        if norm_v > tol:
            idx.append(j)
            basis.append(v / norm_v)
            if len(idx) == h11:   # got enough
                break

    indices = np.array(idx)
    return indices, GLSM_sorted[:,indices]







def axion_spectra_hierarchy(logLambda4, Kmetric, glsm_indp, tol=1e-2):
    """
    **Description:**
    Compute approximate axion masses and decay constants given
    logLambda4 (natural), Kmetric, and independent GLSM charges.
    
    Args:
        logLambda4: array-like of log(Λ^4) in natural units.
        Kmetric: 2D numpy array (symmetric positive definite).
        glsm_indp: 2D numpy array of independent GLSM charges (columns).
        tol: float tolerance for error ε_j.
        
    Returns:
        log10ma_approx: numpy array of approximate log10 masses.
        log10fa_approx: numpy array of approximate log10 decay constants.
        eps: numpy array of errors ε_j.
        ok: boolean numpy array indicating if ε_j <= tol.
        
        
    Inputs
    ------
    logLambda4 : (n,) array with natural logs of Λ_j^4
    Kmetric    : (n,n) SPD Kähler metric
    glsm_indp  : (n,n) independent GLSM charges
    tol        : allowed leading-order error ε_j

    Returns
    -------
    log10ma_approx : (n,) log10 of approximate masses m_j ≈ Λ_j^2 * |q_jj|
    log10fa_approx : (n,) log10 of approximate decay constants f_j ≈ 1/|q_jj|
    eps            : (n,) leading-order relative error estimates
    ok             : (n,) boolean mask (True if eps <= tol)
    """
    logLambda4 = np.asarray(logLambda4, dtype=float)
    Kmetric    = np.asarray(Kmetric, dtype=float)
    q0         = np.asarray(glsm_indp, dtype=float)
    n          = logLambda4.size

    # Canonical normalization: q_ini = q0 @ Lk^{-T}
    try:
        Lk = np.linalg.cholesky(Kmetric)   # Kmetric = Lk Lk^T
    except:
        return [], [], [], []
    Y  = np.linalg.solve(Lk, q0)             # columns whitened: Y = Lk^{-1} @ q0
    # QR -> lower-triangular L (Q: orthogonal matrix ; R = upper triangular matrix)
    _, R = np.linalg.qr(Y, mode='reduced')   # QR on Y
    L = R.T
    # tolerant triangular check (avoids false failures from tiny round-off)
    assert np.allclose(L, np.tril(L), atol=1e-12, rtol=0.0)
    
    qdiag  = np.diag(L)
    qdiag_sq = qdiag**2

    # convert logΛ^4 (natural) → log10Λ^4
    log10Lambda4 = logLambda4 / np.log(10.0)

    # errors ε_j
    eps = np.zeros(n)
    for j in range(1,n):
        idx = L[j, :j] != 0
        if not np.any(idx):
            eps[j] = 0.0
            continue
        with np.errstate(divide='ignore', invalid='ignore'):
            logL_sub = np.full(j, -np.inf)
            logL_sub[idx] = np.log10(np.abs(L[j, :j][idx]))
            # note: if qdiag2[k]==0, term -> +inf and eps[j] -> inf (intentional)
            term = (log10Lambda4[j] - log10Lambda4[:j]) + 2*logL_sub - np.log10(qdiag_sq[:j])
        eps[j] = np.sum(10.0**term)

    # masses
    with np.errstate(divide='ignore'):
        log10ma_approx = 0.5*log10Lambda4 + np.log10(np.abs(qdiag)) + np.log10(2*np.pi)
    # decay constants
    with np.errstate(divide='ignore'):
        log10fa_approx = -np.log10(np.abs(qdiag)) - np.log10(2*np.pi)

    ok = eps <= tol
    # match docstring order: (..., eps, ok)
    return log10ma_approx, log10fa_approx, eps, ok


# In this function, I'll do the naive calculation (i.e. assume all prime-torics can host O(1) instantons, with one of them morphable to a gauge-theory instanton)
def axion_spectra(tset: np.ndarray,
                  h11: int,
                  Kijk: np.ndarray,
                  charges: np.ndarray, 
                  cyvol: float,
                  divvols: np.ndarray, 
                  gauge_instanton: bool = True,
                  QCD_div_vol: float = 40,
                  QCD_index_set: Union[int, str] = 'closest',
                  check_quality: bool = True,
                  PQ_quality_ratio: float = 1e-10,
                  W0: float = 1.0,
                  mpmath_switch: bool = False):
    """
    **Description:**
    Compute axion masses and decay constants assuming all divisors
    can host O(1) instantons, with one morphable to a gauge-theory instanton.
    
    Args:
        tset: 1D numpy array of 2-cycle volumes.
        h11: int (number of Kähler moduli).
        Kijk: 3D numpy array of triple intersection numbers.
        charges: 2D numpy array of divisor charges (n_div, h11).
        cyvol: float (Calabi-Yau volume).
        divvols: 1D numpy array of divisor volumes.
        gauge_instanton: bool (whether to include gauge instanton).
        QCD_div_vol: float (target volume for QCD divisor).
        QCD_index_set: int or 'closest' (index of QCD divisor or 'closest').
        check_quality: bool (whether to check PQ quality).
        PQ_quality_ratio: float (maximum allowed ratio for PQ quality).
        W0: float (flux superpotential).
        
    Returns:
        log10ma: numpy array of log10 axion masses (or empty if failed).
        log10fa: numpy array of log10 axion decay constants (or empty if failed).
        qcd_indp_pos: int (position of QCD axion in independent set, -2 if failed).
        divvol_qcd: float (volume of QCD divisor, 0 if failed).
    
    """
    
    Mpl_GeV = 2.4*10**18
    Lambda4QCD_eV = 75.5e6**4 # 1511:02867 - this is really topological susceptibility
    log_Lambda4QCD_Pl = np.log(Lambda4QCD_eV * (Mpl_GeV*1e9)**-4)
    Kdim = K_modulus(cyvol)
    logW0 = np.log(W0)
    logLambda4_naive_ini = (logW0 + Kdim) - 2*np.pi*divvols + logextra(divvols)
        
    # Setting "QCD": Here I'm assuming that the closest volume divisor is wrapped by 4 stacks of D7s to have SO(8) Chan-Paton gauge group.
    if gauge_instanton:
        if QCD_index_set=='closest':
            QCD_index_set = np.argmin(np.abs(divvols - QCD_div_vol))

        logLambda4_naive_ini[QCD_index_set] = log_Lambda4QCD_Pl
        
    order_naive = np.argsort(-logLambda4_naive_ini)
    logLambda4_naive_sorted = logLambda4_naive_ini[order_naive]
    glsm_sorted_naive = charges.T[:, order_naive]
    
    # position of QCD divisor in the sorted basis
    qcd_pos_in_order = int(np.where(order_naive == QCD_index_set)[0][0])
    qcd_vec = glsm_sorted_naive[:, qcd_pos_in_order].astype(float)

    final_idx_naive, glsm_indp_naive = select_independent_columns_fast(glsm_sorted_naive, h11, tol=1e-10)
    logLambda4_naive_sorted_indp = logLambda4_naive_sorted[final_idx_naive]
    
    ## -------------------------- Checking quality ---------------------------- ##
    pq_ok = True
    if check_quality:
        
        # check if it was kept as independent
        qcd_is_dependent = not (qcd_pos_in_order in final_idx_naive)
        
        if qcd_is_dependent:
            return [], [], -2, 0
        else:
            qcd_indp_pos = int(np.where(final_idx_naive == qcd_pos_in_order)[0][0])
            
            dep_idx = np.setdiff1d(np.arange(len(order_naive)), final_idx_naive, assume_unique=True)
            dep_mat = glsm_sorted_naive[:, dep_idx].astype(float)        # shape: (rows, #dependents)
            dots = dep_mat.T @ qcd_vec                                   # shape: (#dependents,)
            support_mask = np.abs(dots) > 0.0                            # >1e-12 if floats
            dep_with_qcd = dep_idx[support_mask]
            
            # --- compute log of sqrt(sum (Λ^4)^2) over the selected dependent columns ---
            if dep_with_qcd.size == 0:
                pq_ok = True
            else:
                log_rest = logLambda4_naive_sorted[dep_with_qcd]         # these are logs of Λ^4
                two_log  = 2.0 * log_rest
                m = np.max(two_log)
                log_sum_sq = m + np.log(np.sum(np.exp(two_log - m)))     # log(sum Λ^8)
                log_num = 0.5 * log_sum_sq                               # log sqrt(sum Λ^8)
                log_ratio = log_num - log_Lambda4QCD_Pl                  # compare to Λ_QCD^4
                pq_ok = (log_ratio <= np.log(PQ_quality_ratio))
            
        if not (pq_ok):
            return [], [], -2, 0
        # Dependents sorted decreasing
        # logLambda4_naive_sorted_depd = np.sort(logLambda4_naive_sorted[dep_idx])[::-1]

    divvol_qcd = divvols[order_naive][final_idx_naive][qcd_indp_pos]
    ## ------------------------- Calculating m and f --------------------------- ##
    Kmetric = get_Kahler_metric(Kijk, tset)
    log10ma, log10fa, errors, ok = axion_spectra_hierarchy(logLambda4_naive_sorted_indp, Kmetric,
                                                           glsm_indp_naive, tol=1e-2)
    
    if np.all(ok) or not mpmath_switch: # This includes the case when Kmetric is not SPD. Will return a null array
        return log10ma, log10fa, qcd_indp_pos, divvol_qcd
    else:
        with mp.workdps(70):
            Kmetric_mp = to_mp_matrix(Kmetric)
            glsm_indp_naive_mp = to_mp_matrix(glsm_indp_naive)
            logLam4_naive_mp = [mpf_safe(x) for x in logLambda4_naive_sorted_indp]

            H_scaled, s_naive = stable_weighted_gram_mp(glsm_indp_naive_mp, logLam4_naive_mp)

            m_scaled, Vn, Un, Ln = diagonalize_lagrangian_mp(Kmetric_mp, H_scaled)
            log10m_naive_mp = log10_masses_from_scaled(m_scaled, s_naive, eps=mp.mpf('1e-80'))

            C_naive = forward_substitute(Ln, glsm_indp_naive_mp)
            Cproj_naive = Un.T * C_naive

            max_logw_naive = max(logLam4_naive_mp) if len(logLam4_naive_mp) else mp.ninf
            w_naive = [mp.e**(lw - max_logw_naive) for lw in logLam4_naive_mp]

            r, J = Cproj_naive.rows, Cproj_naive.cols
            log_m2_naive = mp.matrix(r, 1)
            log_Q_naive  = mp.matrix(r, 1)
            for i in range(r):
                t2 = []; t4 = []
                for j in range(J):
                    cij = Cproj_naive[i, j]
                    if cij == 0: continue
                    lc = safe_log_abs(cij)
                    t2.append(2*lc + mp.log(w_naive[j]))
                    t4.append(4*lc + mp.log(w_naive[j]))
                log_m2_naive[i] = row_logsumexp(t2)
                log_Q_naive[i]  = row_logsumexp(t4)

            logf_naive_mp = mp.matrix(r, 1)
            for i in range(r):
                if log_m2_naive[i] == mp.ninf or log_Q_naive[i] == mp.ninf:
                    logf_naive_mp[i] = mp.ninf
                else:
                    logf_naive_mp[i] = mp.mpf('0.5') * (log_m2_naive[i] - log_Q_naive[i])

            log10ma = mp_to_numpy(log10m_naive_mp)[:, 0] + np.log10(2*np.pi)
            log10fa = mp_to_numpy(logf_naive_mp)[:, 0] / np.log(10.0) - np.log10(2*np.pi)
            return log10ma, log10fa, qcd_indp_pos, divvol_qcd
    