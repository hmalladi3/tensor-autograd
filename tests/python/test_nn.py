"""Unit tests for tensor_autograd.nn — Module, Parameter, layers, losses, init."""

import math
import unittest

from tensor_autograd import (
    DType,
    Tensor,
    manual_seed,
    no_grad,
    randn,
    tensor,
    zeros,
)
from tensor_autograd import nn
from tensor_autograd.nn import functional as F


# --------------------------------------------------------------------------- #
# Parameter                                                                   #
# --------------------------------------------------------------------------- #


class TestParameter(unittest.TestCase):
    def test_parameter_inherits_tensor_data_and_marks_requires_grad(self):
        data = tensor([[1.0, 2.0], [3.0, 4.0]])
        before_shape = data.shape
        p = nn.Parameter(data)
        self.assertTrue(p.requires_grad)
        self.assertEqual(p.shape, before_shape)
        self.assertEqual(p.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_parameter_hollows_out_source_to_prevent_double_decref(self):
        data = tensor([1.0, 2.0, 3.0])
        p = nn.Parameter(data)
        self.assertIsNone(data._handle)
        # del data should not crash even though its handle is None.
        del data
        # p still works.
        self.assertEqual(p.tolist(), [1.0, 2.0, 3.0])


# --------------------------------------------------------------------------- #
# Module: registration + traversal                                            #
# --------------------------------------------------------------------------- #


class TestModuleRegistration(unittest.TestCase):
    def test_assigning_parameter_registers_it(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(zeros(3))

        m = M()
        params = list(m.parameters())
        self.assertEqual(len(params), 1)
        self.assertIs(params[0], m.w)

    def test_assigning_submodule_recurses_into_its_parameters(self):
        class Sub(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(zeros(2))

        class Parent(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = Sub()
                self.b = nn.Parameter(zeros(5))

        p = Parent()
        params = list(p.parameters())
        self.assertEqual(len(params), 2)
        ids = {id(x) for x in params}
        self.assertIn(id(p.sub.w), ids)
        self.assertIn(id(p.b), ids)

    def test_assigning_non_parameter_non_module_does_not_register(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.scalar = 42
                self.string = "hi"

        self.assertEqual(list(M().parameters()), [])

    def test_replacing_a_parameter_with_a_non_parameter_unregisters(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(zeros(3))

        m = M()
        self.assertEqual(len(list(m.parameters())), 1)
        m.w = None
        self.assertEqual(len(list(m.parameters())), 0)

    def test_parameters_deduplicates_by_identity(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                shared = nn.Parameter(zeros(3))
                self.a = shared
                self.b = shared

        # Both a and b point at the same Parameter; should yield once.
        self.assertEqual(len(list(M().parameters())), 1)

    def test_zero_grad_clears_grads_on_every_parameter(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(zeros(2))

        m = M()
        m.w._grad = tensor([1.0, 2.0])
        m.zero_grad()
        self.assertIsNone(m.w._grad)


class TestModuleProtocol(unittest.TestCase):
    def test_call_dispatches_to_forward(self):
        class Double(nn.Module):
            def forward(self, x):
                return x * 2.0

        d = Double()
        self.assertEqual(d(tensor([1.0, 2.0])).tolist(), [2.0, 4.0])

    def test_train_and_eval_toggle_training_flag_recursively(self):
        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = nn.Linear(3, 2)

        m = M()
        self.assertTrue(m.training)
        self.assertTrue(m.inner.training)
        m.eval()
        self.assertFalse(m.training)
        self.assertFalse(m.inner.training)
        m.train()
        self.assertTrue(m.training)
        self.assertTrue(m.inner.training)


# --------------------------------------------------------------------------- #
# Linear                                                                      #
# --------------------------------------------------------------------------- #


class TestLinear(unittest.TestCase):
    def test_linear_output_shape(self):
        lin = nn.Linear(3, 5)
        out = lin(zeros(2, 3))
        self.assertEqual(out.shape, (2, 5))

    def test_linear_with_no_bias_has_no_bias_parameter(self):
        lin = nn.Linear(3, 5, bias=False)
        params = list(lin.parameters())
        self.assertEqual(len(params), 1)  # weight only
        self.assertIsNone(lin.bias)

    def test_linear_bias_starts_at_zero(self):
        lin = nn.Linear(3, 5)
        self.assertEqual(lin.bias.tolist(), [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_linear_weight_within_kaiming_uniform_bound(self):
        manual_seed(0)
        in_f = 10
        lin = nn.Linear(in_f, 20)
        bound = math.sqrt(6.0 / in_f)
        flat = [v for row in lin.weight.tolist() for v in row]
        self.assertTrue(all(-bound <= v < bound for v in flat))


# --------------------------------------------------------------------------- #
# Sequential                                                                  #
# --------------------------------------------------------------------------- #


class TestSequential(unittest.TestCase):
    def test_sequential_threads_input_through_layers(self):
        s = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        out = s(zeros(8, 3))
        self.assertEqual(out.shape, (8, 2))

    def test_sequential_exposes_all_parameters(self):
        s = nn.Sequential(nn.Linear(3, 4), nn.ReLU(), nn.Linear(4, 2))
        # Two Linear layers × (weight + bias) = 4 parameters.
        self.assertEqual(sum(1 for _ in s.parameters()), 4)


# --------------------------------------------------------------------------- #
# Activation modules vs functional                                            #
# --------------------------------------------------------------------------- #


class TestActivations(unittest.TestCase):
    def test_relu_module_matches_functional(self):
        x = tensor([-1.0, 0.0, 2.0])
        self.assertEqual(nn.ReLU()(x).tolist(), F.relu(x).tolist())

    def test_sigmoid_module_matches_functional(self):
        x = tensor([0.0, 1.0])
        a = nn.Sigmoid()(x).tolist()
        b = F.sigmoid(x).tolist()
        for ai, bi in zip(a, b):
            self.assertAlmostEqual(ai, bi, places=6)


# --------------------------------------------------------------------------- #
# Losses                                                                      #
# --------------------------------------------------------------------------- #


class TestLosses(unittest.TestCase):
    def test_mse_loss(self):
        pred = tensor([1.0, 2.0, 3.0])
        target = tensor([2.0, 2.0, 2.0])
        # Mean of (-1, 0, 1)^2 = 2/3
        self.assertAlmostEqual(F.mse_loss(pred, target).item(), 2.0 / 3.0, places=5)

    def test_cross_entropy_matches_hand_computation(self):
        # logits = [[2, 1, 0.1]], target = 0
        # log_softmax(logits)[0] = 2 - logsumexp([2,1,0.1])
        logits = tensor([[2.0, 1.0, 0.1]])
        targets = tensor([0])
        lse = math.log(math.exp(2.0) + math.exp(1.0) + math.exp(0.1))
        expected = -(2.0 - lse)
        self.assertAlmostEqual(F.cross_entropy(logits, targets).item(), expected, places=4)

    def test_cross_entropy_shape_check(self):
        with self.assertRaises(ValueError):
            F.cross_entropy(tensor([1.0, 2.0]), tensor([0]))  # 1-D logits

    def test_cross_entropy_batch_size_mismatch(self):
        with self.assertRaises(ValueError):
            F.cross_entropy(tensor([[1.0, 2.0], [3.0, 4.0]]), tensor([0, 1, 2]))


# --------------------------------------------------------------------------- #
# nn.init                                                                     #
# --------------------------------------------------------------------------- #


class TestInit(unittest.TestCase):
    def test_zeros_and_ones(self):
        self.assertEqual(nn.init.zeros((2, 3)).tolist(), [[0.0] * 3, [0.0] * 3])
        self.assertEqual(nn.init.ones((3,)).tolist(), [1.0, 1.0, 1.0])

    def test_kaiming_uniform_bounds(self):
        manual_seed(0)
        t = nn.init.kaiming_uniform(50, (100,))
        bound = math.sqrt(6.0 / 50)
        self.assertTrue(all(-bound <= v < bound for v in t.tolist()))

    def test_xavier_uniform_bounds(self):
        manual_seed(0)
        t = nn.init.xavier_uniform(50, 50, (200,))
        bound = math.sqrt(6.0 / 100)
        self.assertTrue(all(-bound <= v < bound for v in t.tolist()))


# --------------------------------------------------------------------------- #
# End-to-end: train a tiny classifier                                         #
# --------------------------------------------------------------------------- #


class TestTrainingLoop(unittest.TestCase):
    def test_mlp_separates_two_clusters(self):
        manual_seed(0)
        N = 200
        with no_grad():
            x_pos = randn(N // 2, 2) + tensor([[2.0, 2.0]])
            x_neg = randn(N // 2, 2) + tensor([[-2.0, -2.0]])
            x = tensor(x_pos.tolist() + x_neg.tolist())
            y = tensor([1] * (N // 2) + [0] * (N // 2))

        model = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )
        lr = 0.1
        initial_loss = F.cross_entropy(model(x), y).item()
        for _ in range(60):
            model.zero_grad()
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            with no_grad():
                for p in model.parameters():
                    p._inplace_axpy(p._grad, -lr)

        with no_grad():
            preds = model(x).argmax(axes=1)
            correct = sum(int(a == b) for a, b in zip(preds.tolist(), y.tolist()))
        self.assertGreaterEqual(correct / N, 0.95)
        self.assertLess(F.cross_entropy(model(x), y).item(), initial_loss * 0.1)


if __name__ == "__main__":
    unittest.main()
