# cosolvers

Provides implementations of the Conjugate Orthogonal Conjugate Gradient (COCG) and Conjugate Orthogonal Conjugate Residual (COCR) methods for solving complex symmetric linear systems.

The interface is compatible with scipy.sparse.linalg:

```python
cocg(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None)

cocr(A, b, x0=None, *, rtol=1e-5, atol=0., maxiter=None, M=None, callback=None)
```

## Table of Contents

- [cosolvers](#cosolvers)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Usage](#usage)
  - [License](#license)

## Installation

```console
pip install git+https://github.com/chillenb/cosolvers
```

## Usage
```python
import numpy as np
import scipy
from cosolvers import cocr

A = np.array([
  [2.0+0.1j, -0.5j],
  [-0.5j, 1.0+0.1j]
])

# diagonal preconditioner, optional
M = scipy.sparse.diags_array(1./np.diag(A))

b = np.array([3.+2.j, 2.])

x, info = cocr(A, b, M=M)

assert np.allclose(A@x, b)
```

## License

`cosolvers` is distributed under the terms of the [BSD 3-clause](https://spdx.org/licenses/BSD-3-Clause.html) license.
