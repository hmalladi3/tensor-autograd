"""Unit tests for tensor_autograd.optim."""

import unittest

from tensor_autograd import (
    Tensor,
    manual_seed,
    no_grad,
    randn,
    tensor,
)
from tensor_autograd import nn, optim
from tensor_autograd.nn import functional as F


class TestOptimizerBase(unittest.TestCase):
    def test_init_materializes_iterable_into_list(self):
        gen = (nn.Parameter(tensor([0.0])) for _ in range(3))
        opt = optim.SGD(gen, lr=0.1)
        self.assertEqual(len(opt.params), 3)

    def test_zero_grad_clears_all_param_grads(self):
        p = nn.Parameter(tensor([0.0]))
        p._grad = tensor([1.0])
        opt = optim.SGD([p], lr=0.1)
        opt.zero_grad()
        self.assertIsNone(p._grad)

    def test_step_skips_params_with_none_grad(self):
        p = nn.Parameter(tensor([5.0, 5.0]))
        # No grad ever set — step should not raise nor modify p.
        before = p.tolist()
        optim.SGD([p], lr=0.1).step()
        self.assertEqual(p.tolist(), before)


class TestSGD(unittest.TestCase):
    def test_vanilla_sgd_step_moves_parameter_along_negative_grad(self):
        p = nn.Parameter(tensor([1.0, 2.0]))
        p._grad = tensor([0.1, 0.2])
        optim.SGD([p], lr=1.0).step()
        # p ← p - lr * g = [1, 2] - [0.1, 0.2] = [0.9, 1.8]
        for actual, want in zip(p.tolist(), [0.9, 1.8]):
            self.assertAlmostEqual(actual, want, places=5)

    def test_sgd_with_momentum_accumulates_buffer(self):
        p = nn.Parameter(tensor([10.0]))
        opt = optim.SGD([p], lr=1.0, momentum=0.5)

        # Step 1: g = [1]. buf = g = [1]. p -= 1 * [1] → [9]
        p._grad = tensor([1.0])
        opt.step()
        self.assertAlmostEqual(p.tolist()[0], 9.0, places=5)

        # Step 2: g = [1]. buf = 0.5 * [1] + [1] = [1.5]. p -= 1 * [1.5] → [7.5]
        p._grad = tensor([1.0])
        opt.step()
        self.assertAlmostEqual(p.tolist()[0], 7.5, places=5)

    def test_sgd_preserves_parameter_handle_identity(self):
        p = nn.Parameter(tensor([1.0]))
        handle_before = p._handle
        p._grad = tensor([0.1])
        optim.SGD([p], lr=0.1).step()
        self.assertIs(p._handle, handle_before)

    def test_sgd_step_runs_under_no_grad(self):
        # After a step, the parameter should not have a _grad_fn even if
        # arithmetic ran during the step.
        p = nn.Parameter(tensor([1.0]))
        p._grad = tensor([1.0])
        optim.SGD([p], lr=0.1, momentum=0.5).step()
        self.assertIsNone(p._grad_fn)

    def test_negative_momentum_raises(self):
        with self.assertRaises(ValueError):
            optim.SGD([nn.Parameter(tensor([0.0]))], lr=0.1, momentum=-0.1)


class TestAdam(unittest.TestCase):
    def test_adam_step_moves_parameter_in_negative_grad_direction(self):
        p = nn.Parameter(tensor([1.0]))
        p._grad = tensor([1.0])
        optim.Adam([p], lr=0.1).step()
        # First step: m and v from zero, both inputs equal to g.
        # bias-corrected m_hat ≈ g, v_hat ≈ g^2 → update ≈ sign(g).
        # p ≈ 1 - 0.1 = 0.9
        self.assertAlmostEqual(p.tolist()[0], 0.9, places=2)

    def test_adam_state_persists_across_steps(self):
        p = nn.Parameter(tensor([0.0]))
        opt = optim.Adam([p], lr=0.01)
        for _ in range(5):
            p._grad = tensor([1.0])
            opt.step()
        st = opt.state[id(p)]
        self.assertEqual(st["step"], 5)
        self.assertIsNotNone(st["m"])
        self.assertIsNotNone(st["v"])

    def test_adam_invalid_betas_raise(self):
        with self.assertRaises(ValueError):
            optim.Adam([nn.Parameter(tensor([0.0]))], lr=0.1, betas=(0.9, 1.5))

    def test_adam_invalid_eps_raises(self):
        with self.assertRaises(ValueError):
            optim.Adam([nn.Parameter(tensor([0.0]))], lr=0.1, eps=0.0)


class TestEndToEndTraining(unittest.TestCase):
    def test_sgd_with_momentum_trains_an_mlp(self):
        manual_seed(0)
        N = 200
        with no_grad():
            x_pos = randn(N // 2, 2) + tensor([[2.0, 2.0]])
            x_neg = randn(N // 2, 2) + tensor([[-2.0, -2.0]])
            x = tensor(x_pos.tolist() + x_neg.tolist())
            y = tensor([1] * (N // 2) + [0] * (N // 2))

        model = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 2))
        opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        initial = F.cross_entropy(model(x), y).item()
        for _ in range(40):
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        final = F.cross_entropy(model(x), y).item()
        self.assertLess(final, initial * 0.1)

    def test_adam_trains_an_mlp(self):
        manual_seed(1)
        N = 100
        with no_grad():
            x_pos = randn(N // 2, 2) + tensor([[1.5, 1.5]])
            x_neg = randn(N // 2, 2) + tensor([[-1.5, -1.5]])
            x = tensor(x_pos.tolist() + x_neg.tolist())
            y = tensor([1] * (N // 2) + [0] * (N // 2))

        model = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 2))
        opt = optim.Adam(model.parameters(), lr=0.05)

        initial = F.cross_entropy(model(x), y).item()
        for _ in range(60):
            opt.zero_grad()
            F.cross_entropy(model(x), y).backward()
            opt.step()
        final = F.cross_entropy(model(x), y).item()
        self.assertLess(final, initial * 0.2)


if __name__ == "__main__":
    unittest.main()
