# Linear systems methods

This folder contains runnable implementations for the numerical-methods homework:

- LU decomposition with partial pivoting (`P @ A = L @ U`) and a direct solver;
- Jacobi and Gauss-Seidel iterations, returning the solution, convergence status, and residual `A @ x - b`;
- the Thomas algorithm for tridiagonal systems, using only the three diagonals;
- a perturbed-right-hand-side example and a float64 Hilbert-matrix accuracy experiment.

## Run

From the repository root, install the sole dependency and run the examples:

```powershell
py -3 -m pip install -r numerical_methods/requirements.txt
py -3 numerical_methods/linear_systems.py
py -3 -m unittest discover -s numerical_methods -p "test_*.py"
```

Add `--hilbert` to run the dense solve experiment for sizes `n = 100, 200, ..., 1000`:

```powershell
py -3 numerical_methods/linear_systems.py --hilbert
```

The Hilbert experiment constructs `A[i,j] = 1/(i+j+1)` in zero-based indexing, sets the known solution to the all-ones vector, and reports relative forward error, relative residual, and normwise backward error. It uses standard double-precision arithmetic; Hilbert matrices become so ill-conditioned that forward accuracy can be lost even when the residual is small. The output measures that limitation rather than asserting the large systems have accurate solutions. Jacobi and Gauss-Seidel also report `converged=False` if the residual tolerance is not reached within the iteration limit.
