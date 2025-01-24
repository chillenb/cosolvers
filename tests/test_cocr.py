import numpy as np
from scipy.sparse.linalg import LinearOperator

def test_cocr():
    A = np.random.random((10,10)) + 1j*np.random.random((10,10))
    A = A + A.T
    A += np.eye(10)
    from cosolvers.cocr import cocr
    b = np.random.random(10)
    x, info = cocr(A, b, maxiter=200)
    #assert info == 0
    err = np.linalg.norm(A @ x - b)
    assert err < 1e-5
