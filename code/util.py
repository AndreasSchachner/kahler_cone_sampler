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
import os, sys, time, io, pickle, math

from itertools import product, combinations
import numpy as np
from mpmath import mp

# To load pickle files
import pickle
import gzip

def load_zipped_pickle(filen):
    r"""
    **Description:**
    Returns content of zipped pickle file.
    
    
    Args:
       filen (string): Filename of zipped file to be read.
        
    Returns:
       ArrayLike/dict: Data contained in file.
    
    """
    
    with gzip.open(filen, 'rb') as f:
        loaded_object = pickle.load(f)
            
    f.close()
            
    return loaded_object

def save_zipped_pickle(obj, filen, protocol=-1):
    r"""
    **Description:**
    Saves data in a zipped pickle file.
    
    
    Args:
       obj (array/dict): Data to be stored in file.
       filen (string): Filename of file to be read.
        
    Returns:
        
    
    """
    with gzip.open(filen, 'wb') as f:
        pickle.dump(obj, f, protocol)
        
    f.close()

# ---- Precision control (set as you like) ----
mp.dps = 70   # decimal digits

# ---- Utilities ----
def mpf_safe(x):
    """
    **Description:**
    Robustly convert Python/NumPy scalars to mp.mpf without losing precision.
    
    Args:
        x: A scalar value (int, float, str, Decimal, numpy types).
        
    Returns:
        mp.mpf representation of x.
    
    """
    try:
        return mp.mpf(x)            # works for plain int/float/str/Decimal
    except (TypeError, ValueError):
        return mp.mpf(str(x))       # works for numpy.longdouble, numpy.int64, etc.

def to_mp_matrix(A):
    """
    **Description:**
    Convert array-like to mp.matrix with mpf entries, handling NumPy dtypes.
    
    Args:
        A: 2D array-like (list of lists, numpy array, etc.)
    
    Returns:
        mp.matrix with entries converted to mp.mpf.
    
    """
    A = np.asarray(A, dtype=object)           # avoid implicit float64 downcast
    r, c = A.shape
    M = mp.matrix(r, c)
    for i in range(r):
        for j in range(c):
            M[i, j] = mpf_safe(A[i, j])
    return M

def to_mp_vector(x):
    """
    **Description:**
    Convert 1D array-like to mp.matrix vector with mpf entries.
    1D vector -> mp.matrix (n,1) with safe casting.
    
    Args:
        x: 1D array-like (list, numpy array, etc.)
        
    Returns:
        mp.matrix vector with entries converted to mp.mpf.
    
    """
    x = np.asarray(x, dtype=object).ravel()
    n = x.size
    v = mp.matrix(n, 1)
    for i in range(n):
        v[i] = mpf_safe(x[i])
    return v

def zeros_like(A):
    """
    **Description:**
    Create an mp.matrix of zeros with the same shape as A.
    
    Args:
        A: mp.matrix
        
    Returns:
        mp.matrix of zeros with same shape as A.
    """
    
    return mp.matrix(A.rows, A.cols)

def eye(n):
    """
    **Description:**
    Create an n x n identity matrix as mp.matrix.
    
    Args:
        n: Size of the identity matrix.
        
    Returns:
        n x n identity mp.matrix.
        
    """
    
    I = mp.matrix(n, n)
    for i in range(n):
        I[i,i] = mp.one
    return I

def mp_abs(x):  
    """
    **Description:**
    Absolute value function for mp.mpf numbers.
    
    Args:
        x: mp.mpf number.
        
    Returns:
        Absolute value of x as mp.mpf.
    """
    # tiny helper avoids repeated mpfabs calls
    return x if x >= 0 else -x


def row_logsumexp(v):
    """
    **Description:**
    Stable log(sum(exp(v))) for a finite iterable v using mpmath.
    Works with lists, numpy arrays, and mp.mpf entries.
    
    Args:
        v: Iterable of numbers (list, numpy array, etc.)
        
    Returns:
        log(sum(exp(v))) as mp.mpf.
    """
    # Materialize as mp.mpf and handle empty input
    vv = [mp.mpf(x) for x in v]
    if not vv:
        return mp.ninf

    # Max trick for numerical stability
    m = max(vv)
    if m == mp.ninf:           # all entries were -inf
        return mp.ninf

    # Finite, compensated sum
    s = mp.fsum(mp.e**(x - m) for x in vv)
    return m + mp.log(s)

def safe_log_abs(x):
    """
    **Description:**
    Compute log|x| safely, returning -inf if x is zero.
    
    Args:
        x: mp.mpf number.
        
    Returns:
        log|x| as mp.mpf, or -inf if x is zero.
    """
    if x == 0:
        return mp.ninf
    return mp.log(mp_abs(x))


def symmetrize(A):
    """
    **Description:**
    Symmetrize a square mp.matrix A by averaging with its transpose.
    
    Args:
        A: Square mp.matrix.
        
    Returns:
        Symmetric mp.matrix S = (A + A^T) / 2.
    """
    n = A.rows
    S = mp.matrix(n, n)
    for i in range(n):
        for j in range(i, n):
            v = (A[i, j] + A[j, i]) / 2
            S[i, j] = v
            S[j, i] = v
    return S

def first_independent_rows(A):
    """
    **Description:**
    Select the first set of linearly independent rows from matrix A.
    
    Args:
        A: 2D numpy array
        
    Returns:
        numpy array of indices of the selected rows.
    """
    
    n = A.shape[1]
    chosen, current = [], []
    for i in range(A.shape[0]):
        trial = current + [A[i]]
        if np.linalg.matrix_rank(np.vstack(trial)) > len(current):
            chosen.append(i)
            current.append(A[i])
            if len(chosen) == n:
                break
    return np.array(chosen, dtype=int)


### --------------------------------------------------------------------------------------------- ###
### ------------------------------------ Helper functions --------------------------------------- ###
### --------------------------------------------------------------------------------------------- ###    


    
def shuffle_rows(A):
    idx = np.random.permutation(A.shape[0])
    return A[idx], idx

# ---- RANDOM PHASE with EARLY HANDOFF ----
def random_prune_positive_cone_rows(A, h11plus, max_tries=20, comb_limit=200_000, tol=1e-12):
    """
    Random pruning toward h11plus. After ANY successful prune, if comb(N,n) <= comb_limit,
    stop immediately and signal an early deterministic handoff.
    Returns: A_reduced, row_ids, tried_bases, handoff (bool)
    """
    A = np.asarray(A, dtype=float)
    n = A.shape[1]
    row_ids = np.arange(len(A))
    tried_bases = set()
    number_rows = A.shape[0]
    if number_rows == h11plus:
        return A, row_ids, tried_bases, False

    i = 0
    while number_rows > h11plus and i < max_tries:
        A, perm = shuffle_rows(A)
        row_ids = row_ids[perm]

        basis_idx = first_independent_rows(A)
        basis_ids = tuple(sorted(row_ids[basis_idx]))
        if basis_ids in tried_bases:
            i += 1; continue

        A_basis = A[basis_idx]
        try:
            # C = A @ A_basis^{-1}
            C = np.linalg.solve(A_basis.T, A.T).T
        except np.linalg.LinAlgError:
            tried_bases.add(basis_ids); i += 1; continue

        nonneg = np.all(C >= -tol, axis=1)
        pos_any = np.any(C >  tol, axis=1)
        mask_positive = nonneg & pos_any
        mask_positive[basis_idx] = False  # do not delete basis rows

        if not np.any(mask_positive):
            tried_bases.add(basis_ids); i += 1; continue

        deleted_ids = set(row_ids[mask_positive])
        keep_mask = ~mask_positive
        A = A[keep_mask]; row_ids = row_ids[keep_mask]
        number_rows = A.shape[0]

        if tried_bases and deleted_ids:
            tried_bases = {sig for sig in tried_bases if deleted_ids.isdisjoint(sig)}

        if number_rows == h11plus:
            return A, row_ids, tried_bases, False

        try:
            combos = math.comb(number_rows, n)
        except ValueError:
            combos = float('inf')
        if combos <= comb_limit:
            return A, row_ids, tried_bases, True

    return A, row_ids, tried_bases, False


# ---- DETERMINISTIC (SKIP already-tried bases) ----
def prune_positive_cone_rows_skip(A, row_ids, skip_bases, tol=1e-12, target=None):
    A = np.asarray(A, dtype=float)
    N, n = A.shape
    keep = np.ones(N, dtype=bool)
    idx = np.arange(N)

    changed = True
    while changed:
        changed = False
        for I in combinations(idx, n):
            if not keep[list(I)].all():
                continue
            sig = tuple(sorted(row_ids[list(I)]))
            if sig in skip_bases:
                continue

            A_basis = A[list(I), :]
            if np.linalg.matrix_rank(A_basis) < n:
                continue

            C = np.linalg.solve(A_basis.T, A.T).T
            nonneg = np.all(C >= -tol, axis=1)
            pos_any = np.any(C >  tol, axis=1)
            redundant = nonneg & pos_any
            redundant[list(I)] = False
            # only delete rows that are still kept
            redundant &= keep
            if np.any(redundant):
                keep[redundant] = False
                changed = True
                if target is not None and keep.sum() <= target:
                    break  # early exit to target size
        # optional: compact A each pass for memory locality
        A = A[keep]; row_ids = row_ids[keep]
        keep = np.ones(A.shape[0], dtype=bool)
        idx = np.arange(A.shape[0])

    return A, row_ids


# ---- HYBRID WRAPPER (uses early handoff) ----
def hybrid_prune(A, h11plus, max_tries=1000, comb_limit=200_000, tol=1e-12):
    A = np.asarray(A, dtype=float)
    A_rand, row_ids, tried_bases, handoff = random_prune_positive_cone_rows(
        A, h11plus, max_tries=max_tries, comb_limit=comb_limit, tol=tol
    )
    if A_rand.shape[0] <= h11plus:
        return A_rand

    N, n = A_rand.shape
    try:
        combos = math.comb(N, n)
    except ValueError:
        combos = float('inf')

    if handoff or combos <= comb_limit:
        A_det, row_ids_det = prune_positive_cone_rows_skip(
            A_rand, row_ids, tried_bases, tol=tol, target=h11plus
        )
        # Trim to target if we overshot
        if A_det.shape[0] > h11plus:
            # (Optional) finish with LP on the residual set, as your collaborator suggested
            pass
        return A_det

    return A_rand







