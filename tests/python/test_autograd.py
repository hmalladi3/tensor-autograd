"""Unit tests for tensor_autograd.autograd — the reverse-mode autodiff layer."""

import math
import unittest

from tensor_autograd import (
    DType,
    Tensor,
    arange,
    full,
    manual_seed,
    no_grad,
    ones,
    randn,
    tensor,
    zeros,
)


def _close(a, b, tol=1e-4):
    """Compare two scalars or nested lists elementwise within tol."""
    if isinstance(a, list):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return abs(a - b) < tol


# --------------------------------------------------------------------------- #
# Tape recording / requires_grad propagation                                  #
# --------------------------------------------------------------------------- #


class TestTapeRecording(unittest.TestCase):
    def test_op_on_leaf_with_requires_grad_records_node(self):
        a = tensor([1.0, 2.0]); a.requires_grad = True
        b = a * 3.0
        self.assertTrue(b.requires_grad)
        self.assertIsNotNone(b._grad_fn)

    def test_op_without_requires_grad_does_not_record(self):
        a = tensor([1.0, 2.0])
        b = tensor([3.0, 4.0])
        c = a * b
        self.assertFalse(c.requires_grad)
        self.assertIsNone(c._grad_fn)

    def test_no_grad_disables_recording(self):
        a = tensor([1.0, 2.0]); a.requires_grad = True
        with no_grad():
            b = a * 3.0
        self.assertFalse(b.requires_grad)
        self.assertIsNone(b._grad_fn)

    def test_no_grad_is_reentrant_and_restores_state(self):
        a = tensor([1.0]); a.requires_grad = True
        outer = a * 2.0
        with no_grad():
            inner = a * 2.0
        outer_after = a * 2.0
        self.assertTrue(outer.requires_grad)
        self.assertFalse(inner.requires_grad)
        self.assertTrue(outer_after.requires_grad)

    def test_non_differentiable_op_does_not_record(self):
        a = tensor([1.0, 2.0]); a.requires_grad = True
        b = tensor([2.0, 2.0])
        eq = a == b
        self.assertFalse(eq.requires_grad)


# --------------------------------------------------------------------------- #
# Simple analytical gradients                                                 #
# --------------------------------------------------------------------------- #


class TestSimpleGradients(unittest.TestCase):
    def test_sum_of_a_times_b_gives_b_for_a_and_a_for_b(self):
        a = tensor([1.0, 2.0, 3.0]); a.requires_grad = True
        b = tensor([10.0, 20.0, 30.0]); b.requires_grad = True
        ((a * b).sum()).backward()
        self.assertEqual(a._grad.tolist(), [10.0, 20.0, 30.0])
        self.assertEqual(b._grad.tolist(), [1.0, 2.0, 3.0])

    def test_add_propagates_grad_to_both_inputs(self):
        a = tensor([1.0]); a.requires_grad = True
        b = tensor([1.0]); b.requires_grad = True
        ((a + b).sum()).backward()
        self.assertEqual(a._grad.tolist(), [1.0])
        self.assertEqual(b._grad.tolist(), [1.0])

    def test_sub_negates_grad_for_right_operand(self):
        a = tensor([1.0]); a.requires_grad = True
        b = tensor([1.0]); b.requires_grad = True
        ((a - b).sum()).backward()
        self.assertEqual(a._grad.tolist(), [1.0])
        self.assertEqual(b._grad.tolist(), [-1.0])

    def test_neg(self):
        a = tensor([3.0]); a.requires_grad = True
        (-a).sum().backward()
        self.assertEqual(a._grad.tolist(), [-1.0])

    def test_relu_gradient_is_one_where_input_positive(self):
        a = tensor([-1.0, 0.0, 2.0]); a.requires_grad = True
        a.relu().sum().backward()
        # Subgradient at 0 picked as 0.
        self.assertEqual(a._grad.tolist(), [0.0, 0.0, 1.0])

    def test_exp_gradient_equals_output(self):
        a = tensor([0.5, 1.0, 2.0]); a.requires_grad = True
        y = a.exp(); y.sum().backward()
        for g, x in zip(a._grad.tolist(), [0.5, 1.0, 2.0]):
            self.assertAlmostEqual(g, math.exp(x), places=4)

    def test_log_gradient_is_one_over_x(self):
        a = tensor([1.0, 2.0, 4.0]); a.requires_grad = True
        a.log().sum().backward()
        for g, x in zip(a._grad.tolist(), [1.0, 2.0, 4.0]):
            self.assertAlmostEqual(g, 1.0 / x, places=5)

    def test_sqrt_gradient(self):
        a = tensor([1.0, 4.0, 9.0]); a.requires_grad = True
        a.sqrt().sum().backward()
        for g, x in zip(a._grad.tolist(), [1.0, 4.0, 9.0]):
            self.assertAlmostEqual(g, 1.0 / (2.0 * math.sqrt(x)), places=4)

    def test_sigmoid_gradient(self):
        a = tensor(0.0); a.requires_grad = True
        a.sigmoid().backward()
        self.assertAlmostEqual(a._grad.item(), 0.25, places=5)  # sigmoid(0)*(1-sigmoid(0))

    def test_tanh_gradient_at_zero(self):
        a = tensor(0.0); a.requires_grad = True
        a.tanh().backward()
        self.assertAlmostEqual(a._grad.item(), 1.0, places=5)  # 1 - tanh(0)^2


# --------------------------------------------------------------------------- #
# Broadcasting in backward                                                    #
# --------------------------------------------------------------------------- #


class TestBroadcastingBackward(unittest.TestCase):
    def test_grad_of_scalar_input_to_broadcast_op_sums_out(self):
        # y = sum(a + b) where a is scalar, b is (3,). dy/da = 3.
        a = tensor(2.0); a.requires_grad = True
        b = tensor([1.0, 2.0, 3.0])
        (a + b).sum().backward()
        self.assertEqual(a._grad.item(), 3.0)

    def test_grad_collapses_to_size_one_dim(self):
        # row (1,3) + col (2,1) → (2,3). row's grad is sum over dim 0 with keepdim.
        row = tensor([[1.0, 2.0, 3.0]]); row.requires_grad = True
        col = tensor([[10.0], [20.0]]); col.requires_grad = True
        (row + col).sum().backward()
        self.assertEqual(row._grad.shape, (1, 3))
        self.assertEqual(row._grad.tolist(), [[2.0, 2.0, 2.0]])
        self.assertEqual(col._grad.shape, (2, 1))
        self.assertEqual(col._grad.tolist(), [[3.0], [3.0]])


# --------------------------------------------------------------------------- #
# Matmul backward                                                             #
# --------------------------------------------------------------------------- #


class TestMatmulBackward(unittest.TestCase):
    def test_matmul_grad_shapes_and_values(self):
        # C = A @ B. dC/dA = grad_C @ B.T, dC/dB = A.T @ grad_C.
        A = tensor([[1.0, 2.0], [3.0, 4.0]]); A.requires_grad = True
        B = tensor([[1.0, 0.0], [0.0, 1.0]]); B.requires_grad = True
        (A @ B).sum().backward()
        # grad_C is ones((2,2)); dC/dA = ones @ B.T = [[1,1],[1,1]] (since B is identity)
        self.assertEqual(A._grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])
        # dC/dB = A.T @ ones = [[4,4],[6,6]]
        self.assertEqual(B._grad.tolist(), [[4.0, 4.0], [6.0, 6.0]])


# --------------------------------------------------------------------------- #
# View ops differentiate correctly                                            #
# --------------------------------------------------------------------------- #


class TestViewBackward(unittest.TestCase):
    def test_reshape_backward_restores_input_shape(self):
        a = tensor([[1.0, 2.0], [3.0, 4.0]]); a.requires_grad = True
        a.reshape(4).sum().backward()
        self.assertEqual(a._grad.shape, (2, 2))
        self.assertEqual(a._grad.tolist(), [[1.0, 1.0], [1.0, 1.0]])

    def test_transpose_backward_transposes_back(self):
        a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]); a.requires_grad = True
        a.transpose(0, 1).sum().backward()
        self.assertEqual(a._grad.shape, (2, 3))


# --------------------------------------------------------------------------- #
# Gradient accumulation, backward shape requirements                          #
# --------------------------------------------------------------------------- #


class TestBackwardSemantics(unittest.TestCase):
    def test_backward_on_non_scalar_raises_without_grad_arg(self):
        a = tensor([1.0, 2.0]); a.requires_grad = True
        b = a * 2.0
        with self.assertRaises(RuntimeError):
            b.backward()

    def test_backward_accumulates_into_existing_grad(self):
        a = tensor([1.0]); a.requires_grad = True
        (a * 2.0).sum().backward()
        first = a._grad.tolist()[0]
        # Second backward through a freshly-built graph adds to the existing grad.
        (a * 5.0).sum().backward()
        self.assertAlmostEqual(a._grad.tolist()[0], first + 5.0, places=5)

    def test_backward_on_leaf_without_requires_grad_is_noop(self):
        a = tensor([1.0])  # requires_grad=False
        # b would have no grad_fn — backward should not raise.
        b = a + 1.0
        # b is a non-leaf without grad_fn — backward returns early.
        b.backward()  # no-op


# --------------------------------------------------------------------------- #
# A real composite: training a tiny linear regressor                          #
# --------------------------------------------------------------------------- #


class TestTrainingLoop(unittest.TestCase):
    def test_sgd_on_linear_regression_reduces_loss(self):
        # y = 2x + 3 + noise. Fit a 1-feature linear model via plain SGD.
        manual_seed(0)
        N = 200
        # Build data without autograd.
        with no_grad():
            x = randn(N, 1)
            y_target = x * 2.0 + 3.0

        w = tensor([[0.0]]); w.requires_grad = True
        b = tensor(0.0); b.requires_grad = True
        lr = 0.05

        def loss_fn():
            pred = x @ w + b
            err = pred - y_target
            return (err * err).mean()

        initial_loss = float(loss_fn().item())

        for _ in range(60):
            # Zero grads.
            w._grad = None
            b._grad = None
            loss = loss_fn()
            loss.backward()
            # SGD step in-place (no autograd recording).
            with no_grad():
                w._inplace_axpy(w._grad, -lr)
                b._inplace_axpy(b._grad, -lr)

        final_loss = float(loss_fn().item())
        self.assertLess(final_loss, initial_loss * 0.05)
        # Parameters should have converged near the true values.
        self.assertAlmostEqual(w.tolist()[0][0], 2.0, places=1)
        self.assertAlmostEqual(b.item(), 3.0, places=1)


# --------------------------------------------------------------------------- #
# Numerical gradient check                                                    #
# --------------------------------------------------------------------------- #


class TestNumericalGradient(unittest.TestCase):
    def test_finite_difference_matches_analytical(self):
        # f(a, b) = sum(a * b) + sum(log(a + 1))
        def f(a_vals, b_vals):
            a = tensor(a_vals); b = tensor(b_vals)
            return ((a * b).sum() + (a + 1.0).log().sum()).item()

        a = tensor([1.5, 2.5, 0.5]); a.requires_grad = True
        b = tensor([0.5, -1.0, 2.0]); b.requires_grad = True
        ((a * b).sum() + (a + 1.0).log().sum()).backward()

        eps = 1e-3
        for i in range(3):
            ap, am = [1.5, 2.5, 0.5], [1.5, 2.5, 0.5]
            ap[i] += eps; am[i] -= eps
            numerical = (f(ap, [0.5, -1.0, 2.0]) - f(am, [0.5, -1.0, 2.0])) / (2 * eps)
            self.assertAlmostEqual(a._grad.tolist()[i], numerical, places=2)


if __name__ == "__main__":
    unittest.main()
