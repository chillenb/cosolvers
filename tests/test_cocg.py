import numpy as np
from scipy.sparse.linalg import LinearOperator
from cosolvers import cocg

class Counter:
    def __init__(self):
        self.count = 0
    def __call__(self, x):
        self.count += 1

N = 40

def test_cocg():
    A = 1j*np.random.random((N,N))
    A = A + A.T
    A += 100*np.diag(np.random.random(N)+1j*np.random.random(N))
    b = np.random.random(N)
    b = b / np.linalg.norm(b)
    ctr = Counter()
    x, info = cocg(A, b, maxiter=200, callback=ctr)
    #assert info == 0
    correct = np.linalg.solve(A, b)
    err = np.linalg.norm(x - correct)
    assert err < 1e-5
    print(ctr.count)

def test_cocg_precond():
    A = 1j*np.random.random((N,N))
    A = A + A.T
    A += 100*np.diag(np.random.random(N)+1j*np.random.random(N))
    b = np.random.random(N)
    b = b / np.linalg.norm(b)
    ctr = Counter()
    M = np.diag(1/np.diag(A))
    x, info = cocg(A, b, maxiter=200, M=M, callback=ctr)
    #assert info == 0
    err = np.linalg.norm(A @ x - b)
    assert err < 1e-5
    print(ctr.count)