import numpy as np
from scipy.sparse.linalg import LinearOperator

def test_cocg():
    A = np.random.random((10,10)) + 1j*np.random.random((10,10))
    A = A + A.T
    A += np.eye(10)
    from cosolvers.cocg import cocg
    b = np.random.random(10)
    x, info = cocg(A, b, maxiter=200)
    #assert info == 0
    err = np.linalg.norm(A @ x - b)
    assert err < 1e-5
