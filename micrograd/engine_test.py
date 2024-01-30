import unittest
from micrograd.engine import Value, Dot


def has_const(v, val):
    return v._op == '' and v.data == val


def optimize_one(v):
    if v._op == '+':
        left, right = v.args()
        if left._op == '' and right._op == '':
            v.make_equal_to(Value(left.data+right.data))
            return
        if has_const(left, 0): v.make_equal_to(right); return
        if has_const(right, 0): v.make_equal_to(left); return
        if left._op == '+':
            v.make_equal_to(Value(0, (*left.args(), right), '+'))
            return
        # TODO(max): right +

    if v._op == '*':
        left, right = v.args()
        if left._op == '' and right._op == '':
            v.make_equal_to(Value(left.data*right.data))
            return
        if has_const(left, 0): v.make_equal_to(left); return
        if has_const(right, 0): v.make_equal_to(right); return
        if has_const(left, 1): v.make_equal_to(right); return
        if has_const(right, 1): v.make_equal_to(left); return


def optimize(v):
    topo = v.topo()
    for op in topo:
        optimize_one(op)
    topo = v.find().topo()
    for op in reversed(topo):
        args = op.args()
        if op._op == '+' and all(arg._op == '*' for arg in args):
            left_arr = tuple(arg._prev[0] for arg in args)
            right_arr = tuple(arg._prev[1] for arg in args)
            op.make_equal_to(Dot(left_arr, right_arr))


def dot(l, r):
    return sum(li * ri for li, ri in zip(l, r))


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

    def test_const_fold_mul(self):
        x = Value(2)*3
        self.assertEqual(x.find()._op, '*')
        optimize_one(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 6)

    def test_const_fold_multiple_mul(self):
        x = (Value(1)*2)*(Value(3)*4)
        self.assertEqual(x.find()._op, '*')
        optimize_one(x)
        self.assertEqual(x.find()._op, '*')
        optimize(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 24)

    def test_fold_add_zero(self):
        x = Value(1)+0
        self.assertEqual(x.find()._op, '+')
        optimize_one(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 1)

    def test_fold_mul_zero(self):
        x = Value(1)*0
        self.assertEqual(x.find()._op, '*')
        optimize_one(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 0)

    def test_fold_mul_one(self):
        x = Value(2)*1
        self.assertEqual(x.find()._op, '*')
        optimize_one(x)
        self.assertEqual(x.find()._op, '')
        self.assertEqual(x.find().data, 2)

    def test_sum(self):
        l = [Value(0, (), 'var'), Value(0, (), 'var'), Value(0, (), 'var')]
        x = sum(l)
        optimize(x)
        self.assertEqual(x.find()._op, '+')
        self.assertEqual(set(x.find().args()), set(l))

    def test_dot(self):
        l = [Value(0, (), 'var'), Value(0, (), 'var'), Value(0, (), 'var'),
             Value(0, (), 'var'), Value(0, (), 'var'), Value(0, (), 'var')]
        r = [Value(0, (), 'var'), Value(0, (), 'var'), Value(0, (), 'var'),
             Value(0, (), 'var'), Value(0, (), 'var'), Value(0, (), 'var')]
        x = dot(l, r)
        optimize(x)
        self.assertEqual(x.find()._op, 'dot')
        for arg in x.find().args():
            self.assertEqual(arg._op, 'var')


if __name__ == "__main__":
    unittest.main()
