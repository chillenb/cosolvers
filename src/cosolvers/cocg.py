# Copyright (c) 2001-2002 Enthought, Inc. 2003, SciPy Developers,
# 2025 Christopher Hillenbrand <chillenbrand15@gmail.com>
# All rights reserved.

import warnings
import numpy as np
from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg._isolve.utils import make_system

def _get_atol_rtol(name, b_norm, atol=0., rtol=1e-5):
    """
    A helper function to handle tolerance normalization
    """
    if atol == 'legacy' or atol is None or atol < 0:
        msg = (f"'scipy.sparse.linalg.{name}' called with invalid `atol`={atol}; "
               "if set, `atol` must be a real, non-negative number.")
        raise ValueError(msg)

    atol = max(float(atol), float(rtol) * float(b_norm))

    return atol, rtol

def cocg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None):
    """Use Conjugate Orthogonal Conjugate Gradient iteration to solve ``Ax = b``.

    Parameters
    ----------
    A : {sparse array, ndarray, LinearOperator}
        The complex symmetric N-by-N matrix of the linear system.
        Alternatively, `A` can be a linear operator which can
        produce ``Ax`` using, e.g.,
        ``scipy.sparse.linalg.LinearOperator``.
    b : ndarray
        Right hand side of the linear system. Has shape (N,) or (N,1).
    x0 : ndarray
        Starting guess for the solution.
    rtol, atol : float, optional
        Parameters for the convergence test. For convergence,
        ``norm(b - A @ x) <= max(rtol*norm(b), atol)`` should be satisfied.
        The default is ``atol=0.`` and ``rtol=1e-5``.
    maxiter : integer
        Maximum number of iterations.  Iteration will stop after maxiter
        steps even if the specified tolerance has not been achieved.
    M : {sparse array, ndarray, LinearOperator}
        Preconditioner for `A`. It should approximate the
        inverse of `A` (see Notes). Effective preconditioning dramatically improves the
        rate of convergence, which implies that fewer iterations are needed
        to reach a given error tolerance.
    callback : function
        User-supplied function to call after each iteration.  It is called
        as ``callback(xk)``, where ``xk`` is the current solution vector.

    Returns
    -------
    x : ndarray
        The converged solution.
    info : integer
        Provides convergence information:
            0  : successful exit
            >0 : convergence to tolerance not achieved, number of iterations
            <0 : parameter breakdown

    Notes
    -----
    The preconditioner `M` should be a matrix such that ``M @ A`` has a smaller
    condition number than `A`, see [1]_ .

    References
    ----------
    .. [1] "Preconditioner", Wikipedia, 
           https://en.wikipedia.org/wiki/Preconditioner
    """
    A, M, x, b, postprocess = make_system(A, M, x0, b)
    bnrm2 = np.linalg.norm(b)

    atol, _ = _get_atol_rtol('cocg', bnrm2, atol, rtol)

    if bnrm2 == 0:
        return postprocess(b), 0

    n = len(b)

    if maxiter is None:
        maxiter = n*10

    matvec = A.matvec
    psolve = M.matvec

    innerprod = np.vdot
    bilinearprod = np.dot

    rhotol = np.finfo(x.dtype.char).eps**2

    # Dummy values to initialize vars, silence linter warnings
    p = None

    x = x.copy()
    v = b - matvec(x) if x.any() else b.copy()
    w = psolve(v)
    rho = innerprod(v, w)
    beta_prev = 0.0
    p_prev = np.zeros_like(b)

    for iteration in range(maxiter):
        p = w + beta_prev * p_prev
        u = matvec(p)
        mu = bilinearprod(u, p)

        if np.isclose(mu, 0.0, rhotol):
            # Failure, quit
            return postprocess(x), -1

        alpha = rho / mu
        x = x + alpha * p
        v = v - alpha * u

        v_err = np.linalg.norm(v)
        if v_err < atol:
            return postprocess(x), 0

        w = psolve(v)
        rho_next = bilinearprod(v, w)
        if np.abs(rho_next) < rhotol:
            # Failure, quit
            return postprocess(x), -1
        beta = rho_next / rho

        rho = rho_next
        beta_prev = beta
        p_prev = p

        if callback:
            callback(x)

    else:  # for loop exhausted
        # Return incomplete progress
        return postprocess(x), maxiter
