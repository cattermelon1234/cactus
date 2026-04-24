import unittest

from src.transpile.ops import PRECISION_OP
from src.transpile.ops import SUPPORTED_PRECISIONS
from src.transpile.ops import canonicalize_op
from src.transpile.ops import get_op
from src.transpile.ops import has_op


class CanonicalOpsTests(unittest.TestCase):
    def test_layout_aliases_reduce_to_small_surface(self):
        self.assertEqual(canonicalize_op("reshape"), "view")
        self.assertEqual(canonicalize_op("flatten"), "view")
        self.assertEqual(canonicalize_op("transpose"), "permute")
        self.assertEqual(canonicalize_op("concat"), "cat")
        self.assertEqual(canonicalize_op("cos"), "scalar_cos")
        self.assertEqual(canonicalize_op("sin"), "scalar_sin")

    def test_precision_cast_is_canonical_precision_op(self):
        self.assertEqual(PRECISION_OP, "precision_cast")
        self.assertEqual(get_op("precision_cast").backend_op, "precision_cast")
        self.assertIn("fp16", SUPPORTED_PRECISIONS)
        self.assertIn("fp32", SUPPORTED_PRECISIONS)

    def test_supported_op_lookup_uses_aliases(self):
        self.assertTrue(has_op("scaled_dot_product_attention"))
        self.assertEqual(get_op("scaled_dot_product_attention").name, "attention")
        self.assertEqual(get_op("reshape").name, "view")


if __name__ == "__main__":
    unittest.main()
