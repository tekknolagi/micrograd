import unittest
from micrograd.engine import Value

def optimize_one(v):
    if v._op == '+':
        left, right = v.args()
        if left._op == '' and right._op == '':
            v.make_equal_to(Value(left.data+right.data))
            return
    if v._op == '*':
        left, right = v.args()
        if left._op == '' and right._op == '':
            v.make_equal_to(Value(left.data+right.data))
            return


def optimize(v):
    topo = v.topo()
    for op in topo:
        optimize_one(op)


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

    def test_const_fold_plus(self):
        x = Value(1)+2
        self.assertEqual(x.find()._op, '+')
        optimize_one(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 3)

    def test_const_fold_multiple_plus(self):
        x = (Value(1)+2)+(Value(3)+4)
        self.assertEqual(x.find()._op, '+')
        optimize_one(x)
        self.assertEqual(x.find()._op, '+')
        optimize(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 10)


if __name__ == "__main__":
    unittest.main()
