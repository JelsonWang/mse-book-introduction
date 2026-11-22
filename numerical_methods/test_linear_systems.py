import unittest

import numpy as np

from linear_systems import (
    gauss_seidel,
    hilbert_accuracy,
    jacobi,
    lu_decomposition,
    solve_linear_system,
    solve_tridiagonal,
)


class LinearSystemTests(unittest.TestCase):
    def setUp(self):
        self.matrix = np.array(
            [[10.0, -1.0, 2.0], [-1.0, 11.0, -1.0], [2.0, -1.0, 10.0]]
        )
        self.exact = np.array([1.0, 2.0, -1.0])
        self.rhs = self.matrix @ self.exact

    def test_pivoted_lu_reconstructs_matrix(self):
        matrix = np.array([[0.0, 2.0], [1.0, 3.0]])
        permutation, lower, upper = lu_decomposition(matrix)
        np.testing.assert_allclose(permutation @ matrix, lower @ upper)
        np.testing.assert_allclose(solve_linear_system(matrix, np.array([4.0, 7.0])), [1.0, 2.0])

    def test_singular_matrix_is_rejected(self):
        with self.assertRaises(np.linalg.LinAlgError):
            lu_decomposition(np.array([[1.0, 2.0], [2.0, 4.0]]))

    def test_jacobi_and_gauss_seidel_solve_and_report_residual(self):
        for method in (jacobi, gauss_seidel):
            with self.subTest(method=method.__name__):
                result = method(self.matrix, self.rhs)
                self.assertTrue(result.converged)
                np.testing.assert_allclose(result.x, self.exact, atol=1e-9)
                np.testing.assert_allclose(result.residual, self.matrix @ result.x - self.rhs)

    def test_nonconvergent_iteration_is_reported(self):
        result = jacobi(
            np.array([[1.0, 2.0], [2.0, 1.0]]),
            np.array([1.0, 1.0]),
            max_iterations=5,
        )
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 5)

    def test_thomas_solver_matches_dense_solution(self):
        lower = np.array([-1.0, -1.0])
        diagonal = np.array([4.0, 4.0, 4.0])
        upper = np.array([-1.0, -1.0])
        rhs = np.array([4.0, 4.0, 10.0])
        dense = np.diag(diagonal) + np.diag(lower, -1) + np.diag(upper, 1)
        np.testing.assert_allclose(solve_tridiagonal(lower, diagonal, upper, rhs), np.linalg.solve(dense, rhs))
        np.testing.assert_allclose(
            solve_tridiagonal(lower, diagonal, upper, np.array([2.0, 4.0, 10.0])),
            [1.0, 2.0, 3.0],
        )

    def test_thomas_solver_handles_single_equation(self):
        np.testing.assert_allclose(
            solve_tridiagonal(np.array([]), np.array([2.0]), np.array([]), np.array([6.0])),
            [3.0],
        )

    def test_hilbert_accuracy_reports_errors(self):
        size, forward, residual, backward = hilbert_accuracy([8])[0]
        self.assertEqual(size, 8)
        self.assertTrue(np.isfinite(forward))
        self.assertTrue(np.isfinite(residual))
        self.assertTrue(np.isfinite(backward))


if __name__ == "__main__":
    unittest.main()
