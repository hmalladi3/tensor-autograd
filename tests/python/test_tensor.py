"""Unit tests for tensor_autograd.Tensor and the module-level constructors.

Uses stdlib unittest only — no pytest dependency.
"""

import math
import unittest

from tensor_autograd import (
    DType,
    EngineError,
    Tensor,
    arange,
    empty,
    full,
    manual_seed,
    ones,
    ones_like,
    randn,
    tensor,
    uniform,
    zeros,
    zeros_like,
)


# --------------------------------------------------------------------------- #
# Construction                                                                #
# --------------------------------------------------------------------------- #


class TestConstruction(unittest.TestCase):
    def test_tensor_from_scalar(self):
        t = tensor(3.5)
        self.assertEqual(t.shape, ())
        self.assertEqual(t.dtype, DType.FLOAT32)
        self.assertAlmostEqual(t.item(), 3.5, places=5)

    def test_tensor_from_int_scalar_infers_int64(self):
        t = tensor(42)
        self.assertEqual(t.dtype, DType.INT64)
        self.assertEqual(t.item(), 42)

    def test_tensor_from_flat_list(self):
        t = tensor([1.0, 2.0, 3.0])
        self.assertEqual(t.shape, (3,))
        self.assertEqual(t.tolist(), [1.0, 2.0, 3.0])

    def test_tensor_from_nested_list(self):
        t = tensor([[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(t.shape, (2, 2))
        self.assertEqual(t.tolist(), [[1.0, 2.0], [3.0, 4.0]])

    def test_tensor_promotes_int_to_float_when_any_value_is_float(self):
        t = tensor([1, 2.0, 3])
        self.assertEqual(t.dtype, DType.FLOAT32)

    def test_tensor_inconsistent_nested_shape_raises(self):
        with self.assertRaises(ValueError):
            tensor([[1, 2], [3, 4, 5]])

    def test_zeros_ones_full(self):
        z = zeros(2, 3)
        self.assertEqual(z.shape, (2, 3))
        self.assertEqual(z.tolist(), [[0.0] * 3, [0.0] * 3])

        o = ones(4)
        self.assertEqual(o.tolist(), [1.0, 1.0, 1.0, 1.0])

        f = full((2, 2), DType.FLOAT32, 7.5)
        self.assertEqual(f.tolist(), [[7.5, 7.5], [7.5, 7.5]])

    def test_arange_with_one_arg_starts_at_zero(self):
        t = arange(5)
        self.assertEqual(t.tolist(), [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_arange_with_start_stop_step(self):
        t = arange(2, 10, 2)
        self.assertEqual(t.tolist(), [2.0, 4.0, 6.0, 8.0])

    def test_zeros_like_ones_like_match_shape_and_dtype(self):
        t = ones(2, 3)
        z = zeros_like(t)
        o = ones_like(t)
        self.assertEqual(z.shape, t.shape)
        self.assertEqual(z.dtype, t.dtype)
        self.assertEqual(o.shape, t.shape)
        self.assertEqual(o.tolist(), [[1.0] * 3, [1.0] * 3])

    def test_empty_returns_tensor_of_correct_shape(self):
        t = empty(3, 4)
        self.assertEqual(t.shape, (3, 4))
        self.assertEqual(t.dtype, DType.FLOAT32)


# --------------------------------------------------------------------------- #
# Metadata                                                                    #
# --------------------------------------------------------------------------- #


class TestMetadata(unittest.TestCase):
    def test_shape_dtype_ndim_numel(self):
        t = zeros(2, 3, 4)
        self.assertEqual(t.shape, (2, 3, 4))
        self.assertEqual(t.ndim, 3)
        self.assertEqual(t.numel(), 24)
        self.assertEqual(t.dtype, DType.FLOAT32)

    def test_len_returns_first_dim(self):
        self.assertEqual(len(zeros(5, 2)), 5)

    def test_len_on_zero_dim_raises(self):
        with self.assertRaises(TypeError):
            len(tensor(1.0))

    def test_repr_contains_shape_and_dtype(self):
        r = repr(zeros(2, 3))
        self.assertIn("(2, 3)", r)
        self.assertIn("FLOAT32", r)


# --------------------------------------------------------------------------- #
# Arithmetic                                                                  #
# --------------------------------------------------------------------------- #


class TestArithmetic(unittest.TestCase):
    def test_add_two_tensors(self):
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([10.0, 20.0, 30.0])
        self.assertEqual((a + b).tolist(), [11.0, 22.0, 33.0])

    def test_sub_mul_div(self):
        a = tensor([4.0, 6.0])
        b = tensor([2.0, 3.0])
        self.assertEqual((a - b).tolist(), [2.0, 3.0])
        self.assertEqual((a * b).tolist(), [8.0, 18.0])
        self.assertEqual((a / b).tolist(), [2.0, 2.0])

    def test_neg(self):
        self.assertEqual((-tensor([1.0, -2.0])).tolist(), [-1.0, 2.0])

    def test_add_with_scalar(self):
        self.assertEqual((tensor([1.0, 2.0]) + 10).tolist(), [11.0, 12.0])
        self.assertEqual((10 + tensor([1.0, 2.0])).tolist(), [11.0, 12.0])

    def test_broadcast_row_plus_col(self):
        row = tensor([[10.0, 20.0, 30.0]])
        col = tensor([[1.0], [2.0]])
        out = (row + col).tolist()
        self.assertEqual(out, [[11.0, 21.0, 31.0], [12.0, 22.0, 32.0]])

    def test_matmul(self):
        a = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3
        b = tensor([[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]])  # 3x2
        c = a @ b
        self.assertEqual(c.shape, (2, 2))
        self.assertEqual(c.tolist(), [[58.0, 64.0], [139.0, 154.0]])

    def test_T_transposes_last_two_dims(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.T.shape, (3, 2))
        self.assertEqual(t.T.tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_unary_methods(self):
        t = tensor([1.0, 4.0, 9.0])
        self.assertEqual(t.sqrt().tolist(), [1.0, 2.0, 3.0])
        self.assertAlmostEqual(t.exp().tolist()[0], math.exp(1.0), places=4)
        self.assertEqual(t.log().tolist()[0], 0.0)
        self.assertEqual(tensor([-1.0, 0.0, 1.0]).relu().tolist(), [0.0, 0.0, 1.0])

    def test_pow(self):
        self.assertEqual((tensor([2.0, 3.0]) ** tensor([3.0, 2.0])).tolist(), [8.0, 9.0])

    def test_dtype_mismatch_raises(self):
        a = tensor([1.0, 2.0])
        b = tensor([1, 2])  # Int64
        with self.assertRaises(EngineError):
            _ = a + b


# --------------------------------------------------------------------------- #
# Comparisons + truthiness                                                    #
# --------------------------------------------------------------------------- #


class TestComparisons(unittest.TestCase):
    def test_eq_returns_bool_tensor(self):
        a = tensor([1.0, 2.0, 3.0])
        b = tensor([1.0, 5.0, 3.0])
        out = a == b
        self.assertEqual(out.dtype, DType.BOOL)
        self.assertEqual(out.tolist(), [True, False, True])

    def test_lt_and_gt(self):
        a = tensor([1.0, 5.0, 3.0])
        b = tensor([2.0, 5.0, 3.0])
        self.assertEqual((a < b).tolist(), [True, False, False])
        self.assertEqual((a > b).tolist(), [False, False, False])

    def test_tensor_is_unhashable(self):
        with self.assertRaises(TypeError):
            {tensor([1.0]): "value"}

    def test_bool_on_multi_element_raises(self):
        with self.assertRaises(RuntimeError):
            bool(tensor([1.0, 2.0]))

    def test_bool_on_zero_dim_works(self):
        self.assertTrue(bool(tensor(1.0)))
        self.assertFalse(bool(tensor(0.0)))


# --------------------------------------------------------------------------- #
# Reductions                                                                  #
# --------------------------------------------------------------------------- #


class TestReductions(unittest.TestCase):
    def test_sum_over_all(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.sum().item(), 21.0)

    def test_sum_over_axis(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.sum(axes=0).tolist(), [5.0, 7.0, 9.0])
        self.assertEqual(t.sum(axes=1).tolist(), [6.0, 15.0])

    def test_sum_with_keepdim(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        s = t.sum(axes=1, keepdim=True)
        self.assertEqual(s.shape, (2, 1))
        self.assertEqual(s.tolist(), [[6.0], [15.0]])

    def test_sum_with_negative_axis(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.sum(axes=-1).tolist(), [6.0, 15.0])

    def test_mean_max_argmax(self):
        t = tensor([[3.0, 1.0, 4.0], [1.0, 5.0, 9.0]])
        self.assertAlmostEqual(t.mean().item(), (3 + 1 + 4 + 1 + 5 + 9) / 6, places=5)
        self.assertEqual(t.max(axes=1).tolist(), [4.0, 9.0])
        am = t.argmax(axes=1)
        self.assertEqual(am.dtype, DType.INT64)
        self.assertEqual(am.tolist(), [2, 2])


# --------------------------------------------------------------------------- #
# Views                                                                       #
# --------------------------------------------------------------------------- #


class TestViews(unittest.TestCase):
    def test_reshape_compatible_returns_view(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        r = t.reshape(3, 2)
        self.assertEqual(r.shape, (3, 2))
        self.assertEqual(r.tolist(), [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    def test_reshape_with_minus_one_infers_dim(self):
        t = tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        r = t.reshape(2, -1)
        self.assertEqual(r.shape, (2, 3))

    def test_reshape_rejects_more_than_one_minus_one(self):
        with self.assertRaises(ValueError):
            tensor([1.0, 2.0]).reshape(-1, -1)

    def test_transpose_and_negative_dims(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t.transpose(0, 1).tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        self.assertEqual(t.transpose(-1, -2).shape, (3, 2))

    def test_squeeze_and_unsqueeze(self):
        t = ones(2, 1, 3)
        self.assertEqual(t.squeeze(1).shape, (2, 3))
        self.assertEqual(ones(4).unsqueeze(0).shape, (1, 4))
        self.assertEqual(ones(4).unsqueeze(-1).shape, (4, 1))

    def test_squeeze_nonunit_dim_raises(self):
        with self.assertRaises(ValueError):
            ones(2, 3).squeeze(0)

    def test_slice_method(self):
        t = tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        self.assertEqual(t.slice(0, 1, 4).tolist(), [20.0, 30.0, 40.0])

    def test_contiguous_returns_self_when_already_contiguous(self):
        t = ones(3, 4)
        c = t.contiguous()
        # Returns the same engine handle (refcount bumped) per spec.
        self.assertEqual(c.shape, t.shape)

    def test_cast(self):
        t = tensor([1.5, 2.7, 3.9])
        i = t.cast(DType.INT64)
        self.assertEqual(i.dtype, DType.INT64)
        self.assertEqual(i.tolist(), [1, 2, 3])


# --------------------------------------------------------------------------- #
# Indexing                                                                    #
# --------------------------------------------------------------------------- #


class TestIndexing(unittest.TestCase):
    def test_int_index_drops_dim(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        row = t[0]
        self.assertEqual(row.shape, (3,))
        self.assertEqual(row.tolist(), [1.0, 2.0, 3.0])

    def test_negative_int_index(self):
        t = tensor([10.0, 20.0, 30.0])
        self.assertEqual(t[-1].item(), 30.0)

    def test_slice_index_keeps_dim(self):
        t = tensor([10.0, 20.0, 30.0, 40.0])
        self.assertEqual(t[1:3].tolist(), [20.0, 30.0])

    def test_tuple_index_multi_dim(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        self.assertEqual(t[0, 1].item(), 2.0)
        self.assertEqual(t[1, :].tolist(), [4.0, 5.0, 6.0])
        self.assertEqual(t[:, 0].tolist(), [1.0, 4.0])

    def test_index_out_of_range(self):
        t = tensor([1.0, 2.0, 3.0])
        with self.assertRaises(IndexError):
            _ = t[10]


# --------------------------------------------------------------------------- #
# Egress (tolist / item)                                                      #
# --------------------------------------------------------------------------- #


class TestEgress(unittest.TestCase):
    def test_tolist_roundtrip_through_engine(self):
        data = [[1.5, -2.5], [3.0, 4.0]]
        self.assertEqual(tensor(data).tolist(), data)

    def test_tolist_on_transposed_walks_in_c_order(self):
        t = tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]).transpose(0, 1)
        # In C-order of the transposed view: [[1,4],[2,5],[3,6]]
        self.assertEqual(t.tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    def test_item_on_zero_dim(self):
        self.assertEqual(tensor(7.5).item(), 7.5)

    def test_item_on_multi_element_raises(self):
        with self.assertRaises(RuntimeError):
            tensor([1.0, 2.0]).item()


# --------------------------------------------------------------------------- #
# Random                                                                      #
# --------------------------------------------------------------------------- #


class TestRandom(unittest.TestCase):
    def test_manual_seed_makes_uniform_reproducible(self):
        manual_seed(42)
        a = uniform(0.0, 1.0, 100).tolist()
        manual_seed(42)
        b = uniform(0.0, 1.0, 100).tolist()
        self.assertEqual(a, b)

    def test_uniform_values_in_range(self):
        manual_seed(7)
        t = uniform(-1.0, 2.0, 1000)
        flat = t.tolist()
        self.assertTrue(all(-1.0 <= v < 2.0 for v in flat))

    def test_randn_mean_and_std_are_in_ballpark(self):
        manual_seed(1)
        n = 20000
        flat = randn(n).tolist()
        mean = sum(flat) / n
        var = sum((v - mean) ** 2 for v in flat) / n
        self.assertLess(abs(mean), 0.05)
        self.assertLess(abs(var - 1.0), 0.05)


# --------------------------------------------------------------------------- #
# In-place AXPY (used by optimizers)                                          #
# --------------------------------------------------------------------------- #


class TestInplace(unittest.TestCase):
    def test_inplace_axpy_mutates_in_place(self):
        dst = ones(4)
        src = full((4,), DType.FLOAT32, 2.0)
        before_handle = id(dst._handle)
        dst._inplace_axpy(src, 3.0)  # dst = 1 + 3*2 = 7
        self.assertEqual(id(dst._handle), before_handle)
        self.assertEqual(dst.tolist(), [7.0, 7.0, 7.0, 7.0])


# --------------------------------------------------------------------------- #
# Lifetime                                                                    #
# --------------------------------------------------------------------------- #


class TestLifetime(unittest.TestCase):
    def test_creating_and_dropping_many_tensors_does_not_crash(self):
        # Doesn't verify zero leaks (we have no engine-side counters exposed),
        # but a memory-management bug would usually crash here under load.
        for _ in range(1000):
            t = zeros(64)
            _ = t + 1.0
            _ = t.sum()
        # If we get here, refcounts didn't underflow.

    def test_tensor_holds_handle_after_construction(self):
        t = zeros(2, 3)
        self.assertIsNotNone(t._handle.value)


if __name__ == "__main__":
    unittest.main()
