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
import os, sys, time, io, pickle, warnings

import cytools as cyt
import numpy as np
import math
import emcee
import itertools
import matplotlib.pyplot as plt
import corner

import datetime
import h5py
import pandas as pd



import emcee as mc
import corner as co
import classy as cl
import numpy as np
import numpy.random as npr
import scipy.stats as sps
import matplotlib.pyplot as plt
import multiprocessing as mp
import cosmology as cosmo

from cytools import Polytope, Cone
from scipy.linalg import null_space, lu_factor, lu_solve, solve_triangular, cho_factor, cho_solve
from scipy.stats import skew, kurtosis
from emcee.moves import MHMove
from emcee.autocorr import AutocorrError
from multiprocessing import Pool
from tqdm.auto import tqdm
from IPython.display import clear_output
from statsmodels.graphics.tsaplots import plot_acf
from contextlib import redirect_stdout
from IPython.display import clear_output


import numpy as np
from typing import Union
import emcee
from emcee.moves import MHMove
from emcee.autocorr import AutocorrError
import multiprocessing as mpp
import math
from mpmath import mp
from itertools import product, combinations
import numpy as np
from scipy.linalg import cho_factor, cho_solve, pinvh


from util import *
from axions import *


### --------------------------------------------------------------------------------------------- ###
### ------------------------------------- SOME LIKELIHOODS -------------------------------------- ###
### --------------------------------------------------------------------------------------------- ###

def loglik_axion_QCD(log10ma, 
					log10fa, 
					qcd_index, 
					divvol_qcd, 
					qcdmassmean: float = 6e-6, 
                    qcdmasssigma: float = 10**(-0.5), 
                    qcdvolmean: float = 40, 
                    qcdvolsigma: float = 5):
    
    
    log10maqcd = log10ma[qcd_index] + np.log10(2.435e18 * 1e9)
    lk1 = -(log10maqcd - np.log10(qcdmassmean))**2 / (2 * np.log10(qcdmasssigma)**2)
    lk2 = -(divvol_qcd - qcdvolmean)**2 / (2 * qcdvolsigma**2)
    lktot = lk1 + lk2
    return lktot if np.isfinite(lktot) else -np.inf

def log_posterior(theta,verbosity=0):
    """Evaluate the log posterior probability at parameter vector theta."""
    #log10_m_ULA_eV, frac_ULA, omega_b, h, A_s, n_s, tau_reio, omega_CDM = theta
    #theta[2] /= 100. #Undo normalisation
    #theta[4] *= 1.e-9
    #log10_m_ULA_eV, frac_ULA, omega_b, h, A_s, n_s, tau_reio, omega_CDM = theta

    #Uniform prior
    theta_upper_limit = np.array([-23., 0.05, 2.5, 0.75, 2.5, 0.99, 0.09, 0.15, 0.0])
    theta_lower_limit = np.array([-27., 0.00, 1.9, 0.60, 1.5, 0.93, 0.01, 0.05, -3.]) #-2. #Lowered limit to account for best Kahler point
    if verbosity>=2:
    	print(theta, theta >= theta_upper_limit, theta <= theta_lower_limit)
    prior_uniform = np.sum(theta >= theta_upper_limit) + np.sum(theta <= theta_lower_limit)
    
    if prior_uniform > 0:
        if verbosity>=2:
            print('Outside uniform prior')
        return -np.inf, None, None, None, None, None, None, None

    #Gaussian prior on tau
    tau_mean = 0.0566 #Planck Sroll2 low-ell polarisation
    tau_stddev = 0.0058
    log_prior_Gaussian = np.log(sps.norm.pdf(theta[6], tau_mean, tau_stddev))

    log10_m_ULA_eV, frac_ULA, omega_b, h, A_s, n_s, tau_reio, omega_CDM, log10_f_Planck = theta
    omega_b /= 100. #Undo normalisation
    A_s *= 1.e-9

    mass_Hubble_eV = 2.13e-33
    mass_Planck_eV = 2.4e27 #Reduced Planck
    #f = 0.1 * mass_Planck_eV

    cosmo_object = cosmo.CosmologyCalculations((10.**log10_m_ULA_eV)/mass_Hubble_eV, 10.**log10_f_Planck, 1.-frac_ULA, omega_b=omega_b, h=h, A_s=A_s, n_s=n_s, tau_reio=tau_reio, omega_total_DM=omega_CDM/(1.-frac_ULA), mass_Hubble_eV=mass_Hubble_eV, mass_Planck_eV=mass_Planck_eV)
    cosmo_object.set_classy_parameters()
    try:
        if verbosity>=2:
            print('Starting AxiCLASS calculation')
        cosmo_object.create_classy_object()
    except cl.CosmoComputationError:
        if verbosity>=2:
            print('AxiCLASS did not compute')
        return -np.inf, None, None, None, None, None, None, None
    cl_dict = cosmo_object.get_lensed_angular_power_spectra()
    cosmo_object.create_PlanckLitePy_object('/home/cytools/mounted_volume/Software/planck-lite-py')
    cosmo_object.get_matter_power_compression_z_3()

    log_like_Planck = cosmo_object.get_PlanckLitePy_log_likelihood()
    log_like_eBOSS = cosmo_object.get_eBOSS_Lyaf_log_likelihood()
    if verbosity>=2:
    	print('log_like_Planck, log_like_eBOSS, log_prior_tau =', log_like_Planck, log_like_eBOSS, log_prior_Gaussian)

    posterior_output = (log_like_Planck + log_like_eBOSS + log_prior_Gaussian, cosmo_object.delta_lin_2, cosmo_object.n_lin, cosmo_object.classy_object.Omega_m(), cosmo_object.classy_object.sigma8(), cosmo_object.classy_object.S8(), cosmo_object.classy_object.thetai_scf(), cosmo_object.get_phi_initial())
    if verbosity>=2:
    	print('Posterior output =', posterior_output)
    return posterior_output


## ------------------------------- EMCEE autocorrelation helper ------------------------------- ##
def safe_tau_max(sampler, *, window=None, c=5.0, tol=0):
    """
    Returns a finite tau_max (float) or None.
    Optionally compute tau on the last `window` steps to stabilize.
    """
    try:
        if window is not None:
            # Flattened chain shape: (nsteps*nwalkers, ndim)
            chain = sampler.get_chain(flat=False)
            # chain shape: (nsteps, nwalkers, ndim)
            if chain.shape[0] < 2:
                return None
            w = min(window, chain.shape[0])
            # Temporarily monkey-patch sampler._chain slice for autocorr
            # Safer route: use sampler.get_autocorr_time on the whole chain and accept noise
            # Here we compute on full chain; comment the window branch if you prefer.
        taus = sampler.get_autocorr_time(c=c, tol=tol, quiet=True)  # emcee>=3
        tmax = float(np.nanmax(np.asarray(taus, dtype=float)))
        if np.isfinite(tmax) and tmax > 0:
            return tmax
    except AutocorrError:
        pass
    except Exception:
        pass
    return None



# ---------- Optimized initializer using the batched helpers ----------
def init_walkers_dilated_ray_with_checks(
                            nwalkers,
                            tip,                 # (h11,)
                            hyp,                 # (nhyp, h11)
                            Kijk,                # (h11, h11, h11)
                            charges,             # (n_div, h11)  prime-toric -> basis coefficients
                            cyvolmax,            # float
                            lam_min: float = 1.0,
                            lam_max: float = 1e5,
                            divvols_min=1.0,
                            hyp_min=1.0,
                            cyvol_min=1.0,
                            jitter_rel=3e-2,
                            max_tries_per_batch=500,
                            batch_factor=20,
                            orthogonalize_jitter=True,
                            lnlik=False,
                            divvolswitch=False,
                            QCD_index_set='closest',
                            QCD_div_vol=40,
                            gauge_instanton=True,
                            check_quality=True,
                            PQ_quality_ratio=1e-10,
                            W0=1.0,
                            rng=None,
                            mpmath_switch: bool = False
                        ):
    
    rng = np.random.default_rng() if rng is None else rng
    tip = np.asarray(tip, dtype=float)
    hyp = np.asarray(hyp, dtype=float)
    h11 = tip.shape[0]

    tnorm = np.linalg.norm(tip)
    if tnorm == 0:
        raise ValueError("tip must be nonzero")
    if lam_min <= 0:
        raise ValueError("lam_min must be positive (log is undefined at 0)")
    if lam_min >= lam_max:
        raise ValueError("lam_min must be smaller than lam_max")

    u_tip = tip / tnorm
    accepted = []

    current_jitter = jitter_rel
    tries = 0
    needed = nwalkers

    def propose(m, jitter_scale):
        lam = np.exp(rng.uniform(np.log(lam_min), np.log(lam_max), size=m))
        base = lam[:, None] * tip  # (m, h11)

        J = rng.normal(size=(m, h11))
        if orthogonalize_jitter:
            proj = (J @ u_tip)[:, None] * u_tip
            J = J - proj
        J /= np.maximum(np.linalg.norm(J, axis=1, keepdims=True), 1e-16)
        jitter = jitter_scale * tnorm * J
        return base + jitter  # (m, h11)

    while needed > 0 and tries < max_tries_per_batch:
        m = batch_factor * needed
        T = propose(m, current_jitter)

        # 1) Hyperplane constraints (vectorized)
        ok_hyp = (hyp @ T.T > hyp_min).all(axis=0)
        
        # 2) Volumes (fully batched)
        vols_cy  = get_cy_volume(Kijk, T)                 # (m,)
        vols_div = get_divisor_volumes(Kijk, T, charges)  # (m, n_div)

        ok_cy  = (vols_cy >= cyvol_min) & (vols_cy <= cyvolmax)
        ok_div = (vols_div > divvols_min).all(axis=1)

        # 3) Divisor volumes cap (if turned on)
        if divvolswitch and np.all(vols_div > 40):
            ok_div_cap = False
        else:
            ok_div_cap = True
        
        ok = ok_hyp & ok_cy & ok_div & ok_div_cap

        # 4) Optional physics-quality check only on survivors
        if lnlik:
            survivors_idx = np.nonzero(ok)[0]
            if survivors_idx.size:
                ok_quality = np.ones_like(ok, dtype=bool)
                for i in survivors_idx:
                    t = T[i]
                    # reuse already-computed volumes for numerical consistency
                    cyvol_i   = vols_cy[i]
                    divvols_i = vols_div[i]
                    try:
                        log10m, log10f, _, _ = axion_spectra(
                                                        t, h11, Kijk, charges, cyvol_i, divvols_i,
                                                        gauge_instanton=gauge_instanton,
                                                        QCD_div_vol=QCD_div_vol,
                                                        QCD_index_set=QCD_index_set,
                                                        check_quality=check_quality,
                                                        PQ_quality_ratio=PQ_quality_ratio,
                                                        W0=W0,
                                                        mpmath_switch=mpmath_switch
                                                        )
                        if len(log10m) == 0:
                            ok_quality[i] = False
                    except Exception:
                        ok_quality[i] = False
                ok &= ok_quality

        good = T[ok]
        if good.shape[0] == 0:
            current_jitter *= 0.5
            batch_factor = min(batch_factor * 2, 100)
            tries += 1
            continue

        take = min(needed, good.shape[0])
        accepted.append(good[:take])
        needed -= take
        tries += 1

    if needed > 0:
        raise RuntimeError(f"Could not find enough valid initial points; missing {needed}.")

    return np.vstack(accepted)


### --------------------------------------------------------------------------------------------- ###
### ------------------------------------ FOR DIRECT SAMPLING ------------------------------------ ###
### --------------------------------------------------------------------------------------------- ###

def log_prob_DS(theta: np.ndarray,
                 h11: int,
                 hyp: np.ndarray,
                 charges: np.ndarray,
                 Kijk: np.ndarray,
                 cyvolmin: float = 1.0,
                 cyvolmax: float = 1e20,
                 hyp_min: float = 1.0,
                 divvols_min: float = 1.0,
                 divvolswitch: bool = False,
                 divvol_any_max: float = 40.0,
                 lnlik: bool = False,
                 include_cosmology: bool = False,
                 gauge_instanton: bool = True,
                 check_quality: bool = True,
                 QCD_div_vol: float = 40,
                 QCD_index_set: Union[int, str] = 'closest',
                 W0: float = 1.0,
                 mpmath_switch: bool = False,
                 verbosity: int = 0) -> float:
    """
    Returns:
      log_measure + log_likelihood_total, or -inf if any check fails.
    """
    if include_cosmology:
        tset = np.exp(theta[:h11])
        cosmo_params = theta[h11:]
    else:
        tset = np.exp(theta)
        
    # Basic sanity
    if not np.all(np.isfinite(tset)):
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf
    
    # 0) Curve volumes proxy
    if np.any(np.dot(hyp,tset) < hyp_min):
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf
        
    t_norm = np.linalg.norm(tset)
    t_hat = tset/t_norm
    
    cyvol_hat = get_cy_volume(Kijk, t_hat)
    divvols_hat = get_divisor_volumes(Kijk, t_hat, charges)

    cyvol = cyvol_hat * t_norm**3
    divvols = divvols_hat * t_norm**2
    
    # Domain cuts (only after finiteness checks)
    if (cyvol <= cyvolmin) or (cyvol >= cyvolmax):
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf

    if np.any(divvols <= divvols_min):
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf
    
    if divvolswitch and np.all(divvols > divvol_any_max):
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf

    # 2) measure term from Kähler metric
    KJ_hat = get_M_matrix(Kijk, t_hat)
    
    KJ_hat = KJ_hat.astype(np.float64) 
    # SPD guard (cheap): require strictly positive min eigenvalue with margin
    w = np.linalg.eigvalsh(KJ_hat / (t_norm**2))
    lam_min = w[0]
    lam_max = w[-1]

    # absolute and relative safety thresholds
    ABS_TOL = 1e-14
    REL_TOL = 1e-12

    if lam_min <= ABS_TOL or lam_min/lam_max < REL_TOL:
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf

    # numerically stable √det via slogdet; reject non-PD cases
    sign, logdet = np.linalg.slogdet(KJ_hat/(t_norm**2))
    if sign <= 0:
        if include_cosmology:
            return -np.inf, None, None, None, None, None, None, None
        else:
            return -np.inf

    log_measure_extra = np.sum(theta)
    
    log_measure = 0.5*logdet + log_measure_extra

    # Compute axion parameters if lnlik = True.
    if lnlik:
        PQ_quality_ratio=1e-10
        # First entries are for the QCD axion
        log10m, log10f, qcd_indp_pos, divvol_qcd = axion_spectra(tset, h11, Kijk, charges, cyvol, divvols,
                                       gauge_instanton=gauge_instanton, QCD_div_vol=QCD_div_vol, 
                                       QCD_index_set=QCD_index_set, check_quality = check_quality, 
                                       PQ_quality_ratio=PQ_quality_ratio, W0=W0,mpmath_switch=mpmath_switch)
        if len(log10m) == 0:
            if include_cosmology:
                return -np.inf, None, None, None, None, None, None, None
            else:
                return -np.inf # PQ quality was poor
        else:
            # insert the likelihood function for axion spectra
            log_likelihood_total = 0.0
            
        #Include cosmology likelihood
        if include_cosmology:
            theta_cosmology = np.concatenate((np.array([log10m + np.log10(Mpl_GeV * 1.e+9),]), cosmo_params, np.array([log10f,])), axis=0) #Check m,f units
            if verbosity>=2:
                print('theta_cosmology =', theta_cosmology)
            #theta_cosmology = np.array([-25., 0.025, 2.2, 0.675, 2., 0.96, 0.0566, 0.12, -1.]) #log10_m_ULA_eV, frac_ULA, omega_b, h, A_s, n_s, tau_reio, omega_CDM, log10_f_Planck
            log_posterior_cosmology_output = log_posterior(theta_cosmology,verbosity=verbosity)
            log_posterior_cosmology = log_posterior_cosmology_output[0]
            if verbosity>=2:
                print('log_posterior_cosmology_output =', log_posterior_cosmology_output, log_posterior_cosmology)
            if np.isfinite(log_posterior_cosmology):
                log_likelihood_total += log_posterior_cosmology
            else:
                if verbosity>=2:
                    print('Cosmology excludes this parameter point')
                return -np.inf, None, None, None, None, None, None, None
            
    else:
        log_likelihood_total = 0.0
        
    out = float(log_measure + log_likelihood_total)
    if include_cosmology:
        if np.isfinite(out):
            try:
                return out, log_posterior_cosmology_output[1], log_posterior_cosmology_output[2], log_posterior_cosmology_output[3], log_posterior_cosmology_output[4], log_posterior_cosmology_output[5], log_posterior_cosmology_output[6], log_posterior_cosmology_output[7]
            except:
                
                return out, None, None, None, None, None, None, None
        else:
            return -np.inf, None, None, None, None, None, None, None
    else:
        return out if np.isfinite(out) else -np.inf


def emcee_wrapper_DS(h11: int,
                     charges: np.ndarray,
                     Kijk: np.ndarray,
                     hyp: np.ndarray,
                     tip: np.ndarray,
                     cyvolattip: float,
                     nwalkers: int,
                     hyp_min: float = 1.0,
                     divvols_min: float = 1.0,
                     scale: float = 50,
                     burnin: float = 50,
                     cyvolmaxfactor: float = 1e14,                     
                     lnlik: bool = False,
                     include_cosmology: bool = False,
                     gauge_instanton: bool = True,
                     check_quality: bool = True,
                     QCD_div_vol: float = 40,
                     divvol_any_max: float = 40.0,
                     divvolswitch: bool = False,
                     QCD_index_set: Union[int, str] = 'closest',
                     thin: bool = False,
                     max_steps: float = 1e6,
                     W0: float = 1.0,
                     parallelization: bool = True,
                     Kahler_ini = None,
                     print_progress: bool = False,
                     mpmath_switch: bool = False,
                     verbosity: int = 0): # Pass a Kahler_ini to start walkers in its neighbourhood
    
    
    if parallelization:
        pool = mpp.Pool()
    else:
        pool = None
    
    # ============================================
    # geometry‐of‐CY bounds
    cyvolmin, cyvolmax = 1.0, cyvolmaxfactor*cyvolattip

    args = (
        h11,
        hyp,
        charges,
        Kijk,
        cyvolmin,
        cyvolmax,
        hyp_min,
        divvols_min,
        divvolswitch,
        divvol_any_max,
        lnlik,
        include_cosmology,
        gauge_instanton,
        check_quality,
        QCD_div_vol,
        QCD_index_set,
        W0,
        mpmath_switch,
        verbosity
    )
    

    # # ─── SET UP EMCEE ─────────────────────────────────────────────────────────────

    # ─── INITIAL POSITIONS (in φ = log alpha space) ──────────────────────────────────

    if include_cosmology:
        tip_pass = Kahler_ini
        if Kahler_ini is None:
            raise ValueError("Please provide input for Kahler_ini!")
        if verbosity>=1:
            print('Starting Kahler point =', tip_pass)
        delta = 1.e-15 #0.01
        lam_min = 1 - delta
        lam_max = 1 + delta
        hyp_min = 0.01
        divvols_min = 1
        t_ini = init_walkers_dilated_ray_with_checks(nwalkers, tip_pass, hyp, Kijk, charges, cyvolmax, 
                                                    lam_min = lam_min, 
                                                    lam_max = lam_max, 
                                                    hyp_min=hyp_min, 
                                                    divvols_min=divvols_min, 
                                                    divvolswitch=divvolswitch,
                                                    jitter_rel=delta, 
                                                    lnlik=False, 
                                                    QCD_index_set='closest',
                                                    mpmath_switch=mpmath_switch)
    else:
        if divvolswitch:
            t_dilated_max = (40/np.max(get_divisor_volumes(Kijk, tip, charges)))**(1/2) # optionally change to have more dilation
        else:
            t_dilated_max = (cyvolmax/cyvolattip)**(1/3)
        t_ini = init_walkers_dilated_ray_with_checks(nwalkers, tip, hyp, Kijk, charges, cyvolmax, 
                                                     lam_min = 1.0, 
                                                     lam_max = t_dilated_max, 
                                                     lnlik=False, 
                                                     divvolswitch=divvolswitch, 
                                                     QCD_index_set='closest',
                                                     mpmath_switch=mpmath_switch)
    
    theta_ini = np.log(t_ini)
    
    # 5) sanity‐check the condition number
    u, s, vh = np.linalg.svd(theta_ini, full_matrices=False)
    cond = s.max() / s.min()
    if print_progress:
        print(f"▶ Initial condition number: {cond:.1e}")
        if cond > 1e4:
            print("⚠️  High condition number—consider increasing `scale`")

    # ─── SET UP & RUN EMCEE ─────────────────────────────────────────────────────

    if include_cosmology:
        ndims = h11 + 7
    else:
        ndims = h11

    sampler = emcee.EnsembleSampler(
        nwalkers, ndims, log_prob_DS, pool=pool, args=args
    )

    if print_progress:
        print("Burn-in…")

    if include_cosmology:
        rng = npr.default_rng()
        #frac_ULA, omega_b, h, A_s, n_s, tau_reio, omega_CDM
        theta_ini_cosmo = rng.multivariate_normal(mean=[0.0146, 2.237, 0.6745, 2.101, 0.9643, 0.0549, 0.2593*(0.6745**2.)], cov=np.diag([0.000014, 0.00014, 0.000041, 0.00031, 0.000039, 0.000076, 0.000056*(0.6745**2.)]), size=nwalkers) 
        
        theta_ini = np.concatenate((theta_ini, theta_ini_cosmo), axis=1) #Check array shapes

        pos, prob, state, blobs = sampler.run_mcmc(theta_ini, burnin, progress=True, skip_initial_state_check=True)
    else:
        pos, prob, state = sampler.run_mcmc(theta_ini, burnin, progress=True, skip_initial_state_check=True)
    sampler.reset()

    if print_progress:
        print("Main chain…")

    initial_chunk = 1000

    # Phase 1: warm-up
    if include_cosmology:
        pos, prob, state, blobs = sampler.run_mcmc(pos, initial_chunk, progress=True, skip_initial_state_check=True)
    else:
        pos, prob, state = sampler.run_mcmc(pos, initial_chunk, progress=True, skip_initial_state_check=True)
    total_steps = initial_chunk
    
    # --- after Phase 1 (your code up to total_steps = initial_chunk) ----------
    last_good_tau = None

    tau_max = safe_tau_max(sampler, tol=0)
    if tau_max is None:
        tau_max = float(initial_chunk)  # conservative fallback
    else:
        last_good_tau = tau_max


    # schedule phase-2
    min_steps   = scale * tau_max
    # always guard int() against NaN and bound it
    def cadence_from_tau(tau, lo=25, hi=5_000):
        if tau is None or not np.isfinite(tau) or tau <= 0:
            return lo
        return int(np.clip(tau, lo, hi))
    check_every = cadence_from_tau(tau_max)
    next_check  = total_steps + check_every

    converged = False
    SHOW_PROGRESS_PHASE2 = False  # keep Phase 1 visible; quiet inner chunks

    while not converged and total_steps < max_steps:
        to_run = next_check - total_steps
        if to_run <= 0:
            to_run = check_every  # safety net
        if include_cosmology:
            pos, prob, state, blobs = sampler.run_mcmc(pos, to_run, progress=SHOW_PROGRESS_PHASE2, skip_initial_state_check=True)
        else:
            pos, prob, state = sampler.run_mcmc(pos, to_run, progress=SHOW_PROGRESS_PHASE2, skip_initial_state_check=True)
        total_steps += to_run

        # Recompute tau only at checkpoints
        tau_now = safe_tau_max(sampler, tol=0)
        if tau_now is not None:
            last_good_tau = tau_now

        # decide which tau to use (prefer fresh, otherwise last good)
        tau_use = tau_now if (tau_now is not None) else last_good_tau
        # optional: also require a minimum length before trusting tau
        if sampler.iteration < 100:
            tau_use = None

        # convergence test
        if (tau_use is not None) and (total_steps > scale * tau_use):
            if print_progress:
                print(f"Converged after {total_steps} steps; τ_max = {tau_use:.1f}")
            converged = True
            break

        # keep cadence ~ once per tau; bounded; never int(NaN)
        check_every = cadence_from_tau(tau_use)
        next_check  = total_steps + check_every
        if print_progress:
            print(f"… {total_steps} steps, τ_max={tau_now if tau_now is not None else np.nan}; "
                  f"using check_every={check_every}")

    if print_progress:
        if not converged:
            print(f"⚠️  Did not converge by {max_steps} steps")

    # ─── EXTRACT & EXPORT ───────────────────────────────────────────────────────
    
    
    if thin:
        thinfactor = int(max(1, tau_max / 2.0)) # ~τ/2 thinning
        samples = sampler.get_chain(thin=thinfactor, flat=True)
        logps   = sampler.get_log_prob(thin=thinfactor, flat=True)
        if include_cosmology:
            derived_params = sampler.get_blobs(thin=thinfactor, flat=True)
    else:
        samples = sampler.get_chain(flat=True)
        logps   = sampler.get_log_prob(flat=True)
        if include_cosmology:
            derived_params = sampler.get_blobs(flat=True)
    
    samples = np.exp(samples)

    if include_cosmology:
        return samples, logps, converged, derived_params
    else:
        return samples, logps, converged


### --------------------------------------------------------------------------------------------- ###
### ---------------------------------- FOR GENERATOR SAMPLING ----------------------------------- ###
### --------------------------------------------------------------------------------------------- ###


def log_prob_GS(theta: np.ndarray,
                tip: np.ndarray,
                h11: int,
                hyp: np.ndarray,
                gen: np.ndarray,
                charges: np.ndarray,
                Kijk: np.ndarray,
                cyvolmin: float,
                cyvolmax: float,
                hyp_min: float = 1.0,
                divvols_min: float = 1.0,
                divvolswitch: bool = False,
                divvol_any_max: float = 40.0,
                lnlik: bool = False,
                include_cosmology: bool = False,
                gauge_instanton: bool = True,
                check_quality: bool = True,
                QCD_div_vol: float = 40,
                QCD_index_set: Union[int, str] = 'closest',
                W0: float = 1.0,
                mpmath_switch: bool = False,
                verbosity: int = 0) -> float:
    """
    Returns:
      log_measure + log_likelihood_total, or -inf if any check fails.
    """
    
    # tset = tip + np.exp(theta) @ gen
    tset = np.exp(theta) @ gen

    # 0) Curve volumes proxy
    if np.any(np.dot(hyp,tset) < hyp_min):
        return -np.inf
        
    t_norm = np.linalg.norm(tset)
    t_hat = tset/t_norm
    
    cyvol_hat = get_cy_volume(Kijk, t_hat)
    divvols_hat = get_divisor_volumes(Kijk, t_hat, charges)

    cyvol = cyvol_hat * t_norm**3
    divvols = divvols_hat * t_norm**2
    
    # Domain cuts (only after finiteness checks)
    if (cyvol <= cyvolmin) or (cyvol >= cyvolmax):
        return -np.inf
    if np.any(divvols <= divvols_min):
        return -np.inf
    
    if divvolswitch and np.all(divvols > divvol_any_max):
        return -np.inf

    # 2) measure term from Kähler metric
    KJ_hat = get_M_matrix(Kijk, t_hat)
    
    KJ_hat = KJ_hat.astype(np.float64) 
    # SPD guard (cheap): require strictly positive min eigenvalue with margin
    w = np.linalg.eigvalsh(KJ_hat / (t_norm**2))
    lam_min = w[0]
    lam_max = w[-1]

    # absolute and relative safety thresholds
    ABS_TOL = 1e-14
    REL_TOL = 1e-12

    if lam_min <= ABS_TOL or lam_min/lam_max < REL_TOL:
        return -np.inf

    # numerically stable √det via slogdet; reject non-PD cases
    sign, logdet = np.linalg.slogdet(KJ_hat/(t_norm**2))
    if sign <= 0:
        return -np.inf

    log_measure = 0.5*logdet + float(np.sum(theta))

    # Compute axion parameters if lnlik = True.
    if lnlik:
        PQ_quality_ratio=1e-10
        # First entries are for the QCD axion
        if QCD_index_set == 'all':
            log_likelihoods_c = -np.inf
            log_likelihood_total = -np.inf
            for i in range(h11+4):
                log10m, log10f, qcd_indp_pos, divvol_qcd = axion_spectra(tset, h11, Kijk, charges, cyvol, divvols,
                                                                     gauge_instanton=gauge_instanton,
                                                                     QCD_div_vol=QCD_div_vol,
                                                                     QCD_index_set=i,
                                                                     check_quality=check_quality,
                                                                     PQ_quality_ratio=PQ_quality_ratio, 
                                                                     W0=W0,
                                                                     mpmath_switch=mpmath_switch)
                if len(log10m) != 0:
                    log_likelihood_total = loglik_axion_QCD(log10m, log10f, qcd_indp_pos, divvol_qcd)
                    if log_likelihood_total > log_likelihoods_c:
                        log_likelihoods_c = log_likelihood_total
        else:
            log10m, log10f, qcd_indp_pos, divvol_qcd = axion_spectra(tset, h11, Kijk, charges, cyvol, divvols,
                                                                     gauge_instanton=gauge_instanton,
                                                                     QCD_div_vol=QCD_div_vol,
                                                                     QCD_index_set=QCD_index_set,
                                                                     check_quality=check_quality,
                                                                     PQ_quality_ratio=PQ_quality_ratio, 
                                                                     W0=W0,
                                                                     mpmath_switch=mpmath_switch)
            if len(log10m) == 0:
                return -np.inf # PQ quality was poor
            else:
                # insert the likelihood function for axion spectra
                log_likelihood_total = loglik_axion_QCD(log10m, log10f, qcd_indp_pos, divvol_qcd)
            
        #Include cosmology likelihood
        if include_cosmology:
            theta_cosmology = np.concatenate((np.array([log10m + np.log10(Mpl_GeV * 1.e+9),]), cosmo_params, np.array([log10f,])), axis=0) #Check m,f units
            if verbosity>=2:
                print('theta_cosmology =', theta_cosmology)
            #theta_cosmology = np.array([-25., 0.025, 2.2, 0.675, 2., 0.96, 0.0566, 0.12, -1.]) #log10_m_ULA_eV, frac_ULA, omega_b, h, A_s, n_s, tau_reio, omega_CDM, log10_f_Planck
            log_posterior_cosmology_output = log_posterior(theta_cosmology,verbosity=verbosity)
            log_posterior_cosmology = log_posterior_cosmology_output[0]
            if verbosity>=2:
                print('log_posterior_cosmology_output =', log_posterior_cosmology_output, log_posterior_cosmology)
            if np.isfinite(log_posterior_cosmology):
                log_likelihood_total += log_posterior_cosmology
            else:
                if verbosity>=2:
                    print('Cosmology excludes this parameter point')
                return -np.inf
            
    else:
        log_likelihood_total = 0.0
        
    out = float(log_measure + log_likelihood_total)
    return out if np.isfinite(out) else -np.inf

    
def emcee_wrapper_GS(h11: int,
                     charges: np.ndarray,
                     Kijk: np.ndarray,
                     hyp: np.ndarray,
                     gen: np.ndarray,
                     tip: np.ndarray,
                     cyvolattip: float,
                     nwalkers: int,
                     scale: float = 50,
                     burnin: float = 50,
                     cyvolmaxfactor: float = 1e10,
                     hyp_min=1.0,
                     divvols_min=1.0,
                     lnlik: bool = False,
                     include_cosmology: bool = False,
                     gauge_instanton: bool = True,
                     check_quality: bool = True,
                     QCD_div_vol: float = 40,
                     divvol_any_max: float = 40.0,
                     divvolswitch: bool = False,
                     QCD_index_set: Union[int, str] = 'closest',
                     thin: bool = False,
                     max_steps: float = 1e6,
                     W0: float = 1.0,
                     parallelization: bool = True,
                     print_progress: bool = False,
                     mpmath_switch: bool = False,
                     verbosity: int = 0):

    if parallelization:
        pool = mpp.Pool()
    else:
        pool = None
    
    # ============================================
    # geometry‐of‐CY bounds
    cyvolmin, cyvolmax = 1.0, cyvolmaxfactor*cyvolattip

    args = (
        tip,
        h11,
        hyp,
        gen,
        charges,
        Kijk,
        cyvolmin,
        cyvolmax,
        hyp_min,
        divvols_min,
        divvolswitch,
        divvol_any_max,
        lnlik,
        include_cosmology,
        gauge_instanton,
        check_quality,
        QCD_div_vol,
        QCD_index_set,
        W0,
        mpmath_switch,
        verbosity
    )

    # # ─── SET UP EMCEE ─────────────────────────────────────────────────────────────

    # ─── INITIAL POSITIONS (in φ = log alpha space) ──────────────────────────────────
    if divvolswitch:
        t_dilated_max = (40/np.max(get_divisor_volumes(Kijk, tip, charges)))**(1/2) # optionally change to have more dilation
    else:
        t_dilated_max = (cyvolmax/cyvolattip)**(1/3)
    t_ini = init_walkers_dilated_ray_with_checks(nwalkers, tip, hyp, Kijk, charges, cyvolmax, lam_min = 1.0, lam_max = t_dilated_max, 
                                                 lnlik=False, divvolswitch=divvolswitch, QCD_index_set='closest',mpmath_switch=mpmath_switch)
    
    delta = (t_ini - tip).T          # shape: (m, nwalkers)
    if gen.shape[0] == gen.shape[1]:  # square (m == r)
        alpha = np.linalg.solve(gen.T, delta)  # (m, nwalkers)
    else:
        # rectangular: least-squares solution to gen.T @ alpha ≈ delta
        # returns (r, nwalkers), matching alpha's shape when gen.T is (m, r)
        alpha, *_ = np.linalg.lstsq(gen.T, delta, rcond=None)
    # Enforcing positivity before log (typical for log-params):
    eps = 1e-12
    alpha_pos = np.clip(alpha, eps, None)

    theta_ini = np.log(alpha_pos.T)  # (nwalkers, m or r)
    
    # 5) sanity‐check the condition number
    u, s, vh = np.linalg.svd(theta_ini, full_matrices=False)
    cond = s.max() / s.min()
    if print_progress:
        print(f"▶ Initial condition number: {cond:.1e}")
        if cond > 1e4:
            print("⚠️  High condition number—consider increasing `scale`")

    # ─── SET UP & RUN EMCEE ─────────────────────────────────────────────────────
    
    ndim = np.shape(gen)[0]
    
    """
    if include_cosmology:
        ndim = h11 + 7
    else:
        ndim = h11
    """
        
    sampler = emcee.EnsembleSampler(
        nwalkers, ndim, log_prob_GS, pool=pool, args=args
    )

    if print_progress:
        print("Burn-in…")
    pos, prob, state = sampler.run_mcmc(theta_ini, burnin, progress=True, skip_initial_state_check=True)
    sampler.reset()

    if print_progress:
        print("Main chain…")

    initial_chunk = 1000

    # Phase 1: warm-up
    pos, prob, state = sampler.run_mcmc(pos, initial_chunk, progress=True, skip_initial_state_check=True)
    total_steps = initial_chunk
    
    # --- after Phase 1 (your code up to total_steps = initial_chunk) ----------
    last_good_tau = None

    tau_max = safe_tau_max(sampler, tol=0)
    if tau_max is None:
        tau_max = float(initial_chunk)  # conservative fallback
    else:
        last_good_tau = tau_max


    # schedule phase-2
    min_steps   = scale * tau_max
    # always guard int() against NaN and bound it
    def cadence_from_tau(tau, lo=25, hi=5_000):
        if tau is None or not np.isfinite(tau) or tau <= 0:
            return lo
        return int(np.clip(tau, lo, hi))
    check_every = cadence_from_tau(tau_max)
    next_check  = total_steps + check_every

    converged = False
    SHOW_PROGRESS_PHASE2 = False  # keep Phase 1 visible; quiet inner chunks

    while not converged and total_steps < max_steps:
        to_run = next_check - total_steps
        if to_run <= 0:
            to_run = check_every  # safety net
        pos, prob, state = sampler.run_mcmc(
            pos, to_run,
            progress=SHOW_PROGRESS_PHASE2,
            skip_initial_state_check=True
        )
        total_steps += to_run

        # Recompute tau only at checkpoints
        tau_now = safe_tau_max(sampler, tol=0)
        if tau_now is not None:
            last_good_tau = tau_now

        # decide which tau to use (prefer fresh, otherwise last good)
        tau_use = tau_now if (tau_now is not None) else last_good_tau
        # optional: also require a minimum length before trusting tau
        if sampler.iteration < 100:
            tau_use = None

        # convergence test
        if (tau_use is not None) and (total_steps > scale * tau_use):
            if print_progress:
                print(f"Converged after {total_steps} steps; τ_max = {tau_use:.1f}")
            converged = True
            break

        # keep cadence ~ once per tau; bounded; never int(NaN)
        check_every = cadence_from_tau(tau_use)
        next_check  = total_steps + check_every
        if print_progress:
            print(f"… {total_steps} steps, τ_max={tau_now if tau_now is not None else np.nan}; "
                  f"using check_every={check_every}")

    if print_progress:
        if not converged:
            print(f"⚠️  Did not converge by {max_steps} steps")

    # ─── EXTRACT & EXPORT ───────────────────────────────────────────────────────
    
    
    if thin:
        thinfactor = int(max(1, tau_max / 2.0)) # ~τ/2 thinning
        samples = sampler.get_chain(thin=thinfactor, flat=True)
        logps   = sampler.get_log_prob(thin=thinfactor, flat=True)
    else:
        samples = sampler.get_chain(flat=True)
        logps   = sampler.get_log_prob(flat=True)
    
    # samples = tip + (np.exp(samples) @ gen)
    samples = (np.exp(samples) @ gen)
        
    return samples, logps, converged


### --------------------------------------------------------------------------------------------- ###
### ---------------------------------- EMCEE WRAPPER FUNCTION ----------------------------------- ###
### --------------------------------------------------------------------------------------------- ###


def emcee_wrapper(cy,
                  burnin = 50,
                  scale = 75,
                  max_steps = 1e6,
                  cyvolmaxfactor = 1e10,
                  generator_sampling = False,
                  lnlik = False,
                  thin = True,
                  include_cosmology = True, 
                  divvolswitch = False, 
                  gauge_instanton = True,
                  check_quality = True,
                  QCD_div_vol = 40,
                  QCD_index_set = 'closest',
                  W0set = 1,
                  nwalkers = None,
                  do_Kcup = True,
                  mpmath_switch: bool = False,
                  Kahler_ini = None,
                  print_progress = False,
                  verbosity = 0):

    h11set = cy.h11()
    if do_Kcup:
        #cy.axion_prep(do_Kcup=True)
        #kc = cy.mori_cone_cap(in_basis=True).dual() # gives the Kahler cup associated with the Calabi-Yau 'cy'
        mcap = cy.mori_cone_cap(in_basis=True)
        xrays = mcap.extremal_rays()
        kc = Cone(rays=xrays).dual()
    else:
        kc = cy.toric_kahler_cone() # gives the Kahler cone associated with the Calabi-Yau 'cy'
        
    tip = kc.tip_of_stretched_cone(1) # t coordinates for the tip of the SKC

    charges = cy.glsm_charge_matrix(include_origin=False).T
    Kijk = cy.intersection_numbers(format='dense', in_basis=True)

    hyp = kc.hyperplanes() # normals to the toric hyperplanes
    hyp_pruned = hybrid_prune(hyp,h11set)
    
    # Reorder the hyperplane normal vectors in the order they appeared in the original hyp
    idxs = [np.where(np.all(np.isclose(hyp, r, atol=1e-12), axis=1))[0][0] for r in hyp_pruned]
    order = np.argsort(idxs)
    hyp_pruned = hyp_pruned[order]
    
    t_start = time.time()

    if generator_sampling:
        cyvolattip = get_cy_volume(Kijk, tip)
        gen = np.array(kc.extremal_rays())  # Array of all extremal generators
        Ng = np.shape(gen)[0]
        if nwalkers is None:
            nwalkers = int(np.ceil(max(10*Ng,20)))     # a good rule is ≥2×ndim walkers
        tsets, logprobs, converged = emcee_wrapper_GS(h11set, 
        										charges, 
        										Kijk, 
        										hyp_pruned, 
        										gen, 
        										tip, 
        										cyvolattip, 
        										nwalkers, 
        										scale=scale, 
        										burnin=burnin, 
					                        	cyvolmaxfactor=cyvolmaxfactor, 
					                        	lnlik=lnlik,
					                        	include_cosmology=include_cosmology, 
					                        	gauge_instanton=gauge_instanton, 
					                        	check_quality=check_quality, 
					                        	QCD_div_vol=QCD_div_vol, 
					                        	divvolswitch=divvolswitch, 
					                        	QCD_index_set=QCD_index_set, 
					                        	thin=thin, 
					                        	max_steps=max_steps, 
					                        	W0=W0set,
                                                mpmath_switch = mpmath_switch,
					                        	print_progress=print_progress,
					                        	verbosity=verbosity)
    else:
        if nwalkers is None:
            nwalkers = int(np.ceil(max(10*h11set,20)))     # a good rule is ≥2×ndim walkers
        ## -- Change basis to make the h11 independent hyperplane normals to be the positive R^n region -- ##
        hyp_basis_ind = first_independent_rows(hyp_pruned)
        hyp_ind = hyp_pruned[hyp_basis_ind]
        LU, piv = lu_factor(hyp_ind)
        I = np.eye(hyp_ind.shape[0])
        hyp_ind_inv = lu_solve((LU, piv), I)
        # hyp_new = np.dot(hyp_pruned,hyp_ind_inv)
        hyp_new = np.dot(hyp,hyp_ind_inv)
        
        Kijk_new = np.einsum('ijk,ia,jb,kc->abc', Kijk, hyp_ind_inv, hyp_ind_inv, hyp_ind_inv, optimize=True)
        charges_new = charges @ hyp_ind.T
        tip_new = np.dot(hyp_ind,tip)
        cyvolattip_new = get_cy_volume(Kijk_new, tip_new)
        
        
        if include_cosmology:

            if Kahler_ini is None:
                Kahler_ini = tip

            Kahler_ini_new = np.dot(hyp_ind,Kahler_ini * 1.05) #Dilate to get right axion mass
            hyp_min=0.01
        else:
            Kahler_ini_new = None
            hyp_min=1.0
        
        emcee_wrapper_output = emcee_wrapper_DS(h11set, 
										    	charges_new, 
										    	Kijk_new, 
										    	hyp_new, 
										    	tip_new, 
				                            	cyvolattip_new, 
				                            	nwalkers, 
				                            	scale=scale, 
				                            	burnin=burnin, 
				                            	cyvolmaxfactor=cyvolmaxfactor, 
				                            	lnlik=lnlik, 
				                            	include_cosmology=include_cosmology,
				                            	gauge_instanton=gauge_instanton, 
				                            	check_quality=check_quality, 
				                            	QCD_div_vol=QCD_div_vol, 
				                            	divvolswitch=divvolswitch, 
				                            	QCD_index_set=QCD_index_set, 
				                            	thin=thin, 
				                            	max_steps=max_steps, 
				                            	W0=W0set,
                                                Kahler_ini=Kahler_ini_new,
                                                mpmath_switch=mpmath_switch,
				                        		print_progress=print_progress,
				                        		verbosity=verbosity)

        if include_cosmology:
            tsets, logprobs, converged, derived_params = emcee_wrapper_output
        else:
            tsets, logprobs, converged = emcee_wrapper_output

        if include_cosmology:
            other = tsets[:,h11set:]
            tsets = tsets[:,:h11set]
        tsets = tsets@hyp_ind_inv.T
    

    # ─── DONE ───────────────────────────────────────────────────────────────────
    t_end = time.time()
    dt    = t_end - t_start

    if verbosity>=1 or print_progress:
        # print in seconds, or format nicely
        print(f"Total runtime: {dt:.1f} seconds for h11 = {h11set}-")
    
        # # ─── EXTRACT & REPORT ───────────────────────────────────────────────────────
        print(f"Collected {tsets.shape[0]} samples.")

    if generator_sampling:
        cyvolmax = cyvolmaxfactor*cyvolattip
    else:
        cyvolmax = cyvolmaxfactor*cyvolattip_new
    
    if include_cosmology and generator_sampling==False:
        out = (converged,cyvolmax,dt,tsets,logprobs,derived_params,other)
    else:
        out = (converged,cyvolmax,dt,tsets,logprobs)

    return out



def emcee_sampler(poly,
                  N = 100,
                  burnin = 50,
                  scale = 75,
                  max_steps = 1e6,
                  cyvolmaxfactor = 1e10,
                  generator_sampling = False,
                  lnlik = False,
                  path_dir = None,
                  thin = True,
                  include_cosmology = True, 
                  divvolswitch = False, 
                  gauge_instanton = True,
                  check_quality = True,
                  QCD_div_vol = 40,
                  QCD_index_set = 'closest',
                  do_Kcup = True,
                  W0set = 1,
                  nwalkers = None,
                  only_converged = True,
                  mpmath_switch: bool = False,
                  print_progress = False,
                  verbosity = 0):

    
    
    clear_output(wait=True)  # <- this clears the notebook cell output

    frsts = poly.ntfe_frsts(N=N*2, triang_method='grow2d')

    nfrsts = len(frsts)

    h11 = poly.h11("N")
    
    if nfrsts<N:
        print(f"Did not find sufficiently many FRSTs. Found {nfrsts}, need {N}.")

    c = 0
    for i in range(nfrsts):
        
        if verbosity>=1:
            print(f"#FRST: ({i+1}/{nfrsts})     #converged chains: {c}    ", flush=True,end="\r") 
            
        t = frsts[i]
        cy = t.get_cy()


        out = emcee_wrapper(cy,
                  burnin = burnin,
                  scale = scale,
                  max_steps = max_steps,
                  cyvolmaxfactor = cyvolmaxfactor,
                  generator_sampling = generator_sampling,
                  lnlik = lnlik,
                  thin = thin,
                  include_cosmology = include_cosmology, 
                  divvolswitch = divvolswitch, 
                  gauge_instanton = gauge_instanton,
                  check_quality = check_quality,
                  QCD_div_vol = QCD_div_vol,
                  QCD_index_set = QCD_index_set,
                  W0set = W0set,
                  do_Kcup = do_Kcup,
                  nwalkers = nwalkers,
                  mpmath_switch=mpmath_switch,
                  print_progress = print_progress,
                  verbosity = verbosity-1)

        if include_cosmology:
            converged,cyvolmax,dt,tsets,logprobs,derived_params,other = out
        else:
            converged,cyvolmax,dt,tsets,logprobs = out

        if only_converged:
            if not converged:
                continue
            
        
        if generator_sampling:
            file_name = f"tsets_gs_num={c}.p"
        else:
            file_name = f"tsets_ds_num={c}.p"
            
        arr = np.column_stack((tsets, logprobs))
        if path_dir is None:
            path_dir = f"./WP_emcee_chains/{h11}/"

        if not os.path.isdir(path_dir):
            os.makedirs(path_dir, exist_ok=True)
        
        out = (poly.points(),t.heights(),converged,cyvolmax,dt,arr)
        save_zipped_pickle(out,path_dir+file_name)

        c += 1
