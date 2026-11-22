"""Linear-system methods for the numerical-methods homework."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable

import numpy as np


def _square_matrix(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("A must be a square matrix")
    if not np.all(np.isfinite(array)):
        raise ValueError("A must contain only finite values")
    return array


def _vector(vector: np.ndarray, size: int, name: str) -> np.ndarray:
    array = np.asarray(vector, dtype=float)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def lu_decomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return P, L, U such that P @ A = L @ U, using partial pivoting."""
    upper = _square_matrix(matrix).copy()
    size = upper.shape[0]
    permutation = np.eye(size)
    lower = np.eye(size)

    for column in range(size):
        pivot_row = column + int(np.argmax(np.abs(upper[column:, column])))
        if upper[pivot_row, column] == 0:
            raise np.linalg.LinAlgError("A is singular")
        if pivot_row != column:
            upper[[column, pivot_row], :] = upper[[pivot_row, column], :]
            permutation[[column, pivot_row], :] = permutation[[pivot_row, column], :]
            if column:
                lower[[column, pivot_row], :column] = lower[[pivot_row, column], :column]

        factors = upper[column + 1 :, column] / upper[column, column]
        lower[column + 1 :, column] = factors
        upper[column + 1 :, column:] -= factors[:, None] * upper[column, column:]
        upper[column + 1 :, column] = 0.0

    return permutation, lower, upper


def solve_linear_system(matrix: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve A x = b using LU decomposition with partial pivoting."""
    permutation, lower, upper = lu_decomposition(matrix)
    b = _vector(rhs, upper.shape[0], "b")
    permuted_b = permutation @ b
    size = b.size

    forward = np.empty(size)
    for row in range(size):
        forward[row] = permuted_b[row] - lower[row, :row] @ forward[:row]

    solution = np.empty(size)
    for row in range(size - 1, -1, -1):
        if upper[row, row] == 0:
            raise np.linalg.LinAlgError("A is singular")
        solution[row] = (
            forward[row] - upper[row, row + 1 :] @ solution[row + 1 :]
        ) / upper[row, row]
    return solution


@dataclass(frozen=True)
class IterativeResult:
    x: np.ndarray
    iterations: int
    residual: np.ndarray
    converged: bool


def _iteration_inputs(
    matrix: np.ndarray,
    rhs: np.ndarray,
    x0: np.ndarray | None,
    max_iterations: int,
    rtol: float,
    atol: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    a = _square_matrix(matrix)
    b = _vector(rhs, a.shape[0], "b")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive")
    if rtol < 0 or atol < 0:
        raise ValueError("rtol and atol must be non-negative")
    diagonal = np.diag(a)
    if np.any(diagonal == 0):
        raise np.linalg.LinAlgError("A must have a non-zero diagonal")
    x = np.zeros_like(b) if x0 is None else _vector(x0, b.size, "x0").copy()
    tolerance = atol + rtol * np.linalg.norm(b, ord=np.inf)
    return a, b, x, tolerance


def jacobi(
    matrix: np.ndarray,
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    max_iterations: int = 10_000,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> IterativeResult:
    """Solve A x = b by Jacobi iteration; convergence is based on ||A x - b||inf."""
    a, b, x, tolerance = _iteration_inputs(
        matrix, rhs, x0, max_iterations, rtol, atol
    )
    diagonal = np.diag(a)
    remainder = a - np.diag(diagonal)
    residual = a @ x - b
    if np.linalg.norm(residual, ord=np.inf) <= tolerance:
        return IterativeResult(x, 0, residual, True)

    for iteration in range(1, max_iterations + 1):
        x = (b - remainder @ x) / diagonal
        residual = a @ x - b
        if np.linalg.norm(residual, ord=np.inf) <= tolerance:
            return IterativeResult(x, iteration, residual, True)
    return IterativeResult(x, max_iterations, residual, False)


def gauss_seidel(
    matrix: np.ndarray,
    rhs: np.ndarray,
    x0: np.ndarray | None = None,
    *,
    max_iterations: int = 10_000,
    rtol: float = 1e-10,
    atol: float = 1e-12,
) -> IterativeResult:
    """Solve A x = b by Gauss-Seidel iteration using the residual as the stop test."""
    a, b, x, tolerance = _iteration_inputs(
        matrix, rhs, x0, max_iterations, rtol, atol
    )
    residual = a @ x - b
    if np.linalg.norm(residual, ord=np.inf) <= tolerance:
        return IterativeResult(x, 0, residual, True)

    for iteration in range(1, max_iterations + 1):
        for row in range(b.size):
            x[row] = (
                b[row]
                - a[row, :row] @ x[:row]
                - a[row, row + 1 :] @ x[row + 1 :]
            ) / a[row, row]
        residual = a @ x - b
        if np.linalg.norm(residual, ord=np.inf) <= tolerance:
            return IterativeResult(x, iteration, residual, True)
    return IterativeResult(x, max_iterations, residual, False)


def solve_tridiagonal(
    lower: np.ndarray,
    diagonal: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a tridiagonal system in O(n) operations using the Thomas algorithm."""
    diagonal = np.asarray(diagonal, dtype=float)
    if diagonal.ndim != 1 or diagonal.size == 0:
        raise ValueError("diagonal must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(diagonal)):
        raise ValueError("diagonal must contain only finite values")
    size = diagonal.size
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if lower.shape != (size - 1,) or upper.shape != (size - 1,):
        raise ValueError("lower and upper must each have length len(diagonal) - 1")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("lower and upper must contain only finite values")
    b = _vector(rhs, size, "b")

    modified_upper = np.empty(max(size - 1, 0))
    modified_rhs = np.empty(size)
    pivot = diagonal[0]
    if pivot == 0:
        raise np.linalg.LinAlgError("zero pivot in Thomas algorithm")
    if size > 1:
        modified_upper[0] = upper[0] / pivot
    modified_rhs[0] = b[0] / pivot

    for row in range(1, size):
        pivot = diagonal[row] - lower[row - 1] * modified_upper[row - 1]
        if pivot == 0:
            raise np.linalg.LinAlgError("zero pivot in Thomas algorithm")
        if row < size - 1:
            modified_upper[row] = upper[row] / pivot
        modified_rhs[row] = (b[row] - lower[row - 1] * modified_rhs[row - 1]) / pivot

    solution = modified_rhs.copy()
    for row in range(size - 2, -1, -1):
        solution[row] -= modified_upper[row] * solution[row + 1]
    return solution


def hilbert_accuracy(sizes: Iterable[int]) -> list[tuple[int, float, float, float]]:
    """Measure float64 forward error, residual, and backward error on Hilbert systems."""
    results = []
    for size in sizes:
        if size < 1:
            raise ValueError("Hilbert matrix sizes must be positive")
        indices = np.arange(1, size + 1, dtype=float)
        matrix = 1.0 / (indices[:, None] + indices[None, :] - 1.0)
        exact_solution = np.ones(size)
        rhs = matrix @ exact_solution
        computed = np.linalg.solve(matrix, rhs)
        residual = matrix @ computed - rhs
        forward_error = np.linalg.norm(computed - exact_solution) / np.linalg.norm(
            exact_solution
        )
        relative_residual = np.linalg.norm(residual) / np.linalg.norm(rhs)
        denominator = (
            np.linalg.norm(matrix, ord=np.inf) * np.linalg.norm(computed, ord=np.inf)
            + np.linalg.norm(rhs, ord=np.inf)
        )
        backward_error = np.linalg.norm(residual, ord=np.inf) / denominator
        results.append((size, forward_error, relative_residual, backward_error))
    return results


def _demo() -> None:
    matrix = np.array([[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]])
    exact = np.array([1.0, 2.0, -1.0])
    rhs = matrix @ exact

    permutation, lower, upper = lu_decomposition(matrix)
    print("LU check ||P A - L U||inf:", np.linalg.norm(permutation @ matrix - lower @ upper, ord=np.inf))
    print("Direct solve:", solve_linear_system(matrix, rhs))

    for method in (jacobi, gauss_seidel):
        result = method(matrix, rhs)
        print(
            f"{method.__name__}: x={result.x}, iterations={result.iterations}, "
            f"||Ax-b||inf={np.linalg.norm(result.residual, ord=np.inf):.3e}, "
            f"converged={result.converged}"
        )

    ill_conditioned = np.array([[1.0, 1.0], [1.0, 1.000001]])
    exact_pair = np.ones(2)
    rhs_pair = ill_conditioned @ exact_pair
    perturbed_rhs = rhs_pair + np.array([0.0, 1e-6])
    base_solution = solve_linear_system(ill_conditioned, rhs_pair)
    perturbed_solution = solve_linear_system(ill_conditioned, perturbed_rhs)
    relative_rhs_change = np.linalg.norm(perturbed_rhs - rhs_pair) / np.linalg.norm(rhs_pair)
    relative_solution_change = np.linalg.norm(perturbed_solution - base_solution) / np.linalg.norm(
        base_solution
    )
    print(
        "Sensitivity example: relative change in b="
        f"{relative_rhs_change:.3e}, relative change in x={relative_solution_change:.3e}"
    )

    lower_diagonal = np.array([-1.0, -1.0])
    main_diagonal = np.array([4.0, 4.0, 4.0])
    upper_diagonal = np.array([-1.0, -1.0])
    tri_solution = np.array([1.0, 2.0, 3.0])
    tri_rhs = np.array([2.0, 4.0, 10.0])
    print(
        "Tridiagonal solve:",
        solve_tridiagonal(lower_diagonal, main_diagonal, upper_diagonal, tri_rhs),
        "(expected:", tri_solution, ")",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hilbert",
        action="store_true",
        help="also measure float64 accuracy for Hilbert matrices of sizes 100 to 1000",
    )
    args = parser.parse_args()
    _demo()
    if args.hilbert:
        print("\nHilbert matrix accuracy (float64; forward error, relative residual, backward error):")
        for size, forward, residual, backward in hilbert_accuracy(range(100, 1001, 100)):
            print(f"n={size:4d}: {forward:.3e}  {residual:.3e}  {backward:.3e}")
        print(
            "Hilbert matrices are severely ill-conditioned: a small residual does not "
            "guarantee an accurate solution."
        )


if __name__ == "__main__":
    main()
