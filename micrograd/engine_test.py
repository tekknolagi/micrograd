import unittest
from micrograd.engine import Value

class Tests(unittest.TestCase):
    def test_const(self):
        x = Value(123)
        y = Value(123)
        self.assertIsNot(x.find(), y.find())
        x.make_equal_to(y)
        self.assertIsNot(x.find(), y.find())

    def test_op(self):
        x = Value(1)+2
        y = Value(3)+4
        self.assertIsNot(x.find(), y.find())
        x.make_equal_to(y)
        self.assertIs(x.find(), y.find())

    def test_chain(self):
        x = Value(0)+1
        y = Value(0)+1
        z = Value(0)+1
        self.assertIsNot(x.find(), y.find())
        self.assertIsNot(x.find(), z.find())
        self.assertIsNot(y.find(), z.find())
        x.make_equal_to(y)
        self.assertIs(x.find(), y.find())
        y.make_equal_to(z)
        self.assertIs(x.find(), z.find())


if __name__ == "__main__":
    unittest.main()
