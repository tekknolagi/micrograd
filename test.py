import collections
import functools
from micrograd.engine import Value, next_id
from micrograd.nn import MLP


OPT_LOG = collections.Counter()


def optimize_one(v):
    if v._op == "+":
        args = v.args()
        if any(arg._op == "+" for arg in args):
            OPT_LOG["flatten_plus"] += 1
            new_args = []
            for arg in args:
                if arg._op == "+":
                    new_args.extend(arg.args())
                else:
                    new_args.append(arg)
            v.make_equal_to(Value(0, tuple(new_args), "+"))
            return True
        if len(args) == 1:
            OPT_LOG["plus_single"] += 1
            v.make_equal_to(args[0])
            return True
    return False


@functools.lru_cache(maxsize=None)
def hashcons_array(vs):
    return Array(vs)


def run_optimize_one(v):
    topo = v.topo()
    changed = False
    for op in topo:
        changed |= optimize_one(op.find())
    return changed


class Array(Value):
    def __init__(self, data):
        super().__init__(0, data, "array")
        self._id = next_id()

    def __repr__(self):
        return f"Array({self._prev})"


class Dot(Value):
    def __init__(self, left, right):
        super().__init__(0, (left, right), "dot")
        assert len(left._prev) == len(right._prev)
        self._id = next_id()

        # TODO(max): Figure out a way to compute this automatically using chain
        # rule.
        def _backward():
            left = self._prev[0].find()
            right = self._prev[1].find()
            for i in range(left._prev):
                left._prev[i].grad += right._prev[i].data * self.grad
                right._prev[i].grad += left._prev[i].data * self.grad

        self._backward = _backward

    def __repr__(self):
        return f"Dot(left={self._left}, right={self._right})"


def optimize(v):
    while changed := run_optimize_one(v):
        pass
    topo = v.find().topo()
    for op in topo:
        args = op.args()
        if op._op == "+" and any(arg._op == "*" for arg in args):
            mul_args = tuple(arg for arg in args if arg._op == "*")
            assert all(len(arg._prev) == 2 for arg in mul_args)
            mul_left = hashcons_array(tuple(arg.arg(0) for arg in mul_args))
            mul_right = hashcons_array(tuple(arg.arg(1) for arg in mul_args))
            other_args = tuple(arg for arg in args if arg._op != "*")
            op.make_equal_to(Value(0, (Dot(mul_left, mul_right), *other_args), "+"))
            changed = True
            continue
    return changed


def fmt(v):
    return f"v{v._id}"


def pretty(v):
    topo = v.topo()
    for op in topo:
        if op._op == "input":
            print(f"{fmt(op)} = input")
        elif op._op == "":
            print(f"{fmt(op)} = {op.data}")
        else:
            print(f"{fmt(op)} = {op._op} {' '.join(fmt(c) for c in op.args())}")


def count(v):
    c = collections.Counter()
    for op in v.topo():
        c[op._op] += 1
    return c


def compile(v):
    for op in v.topo():
        if op._op == "dot":
            n = len(op._prev[0]._prev)
            args = op.args()
            print(f"double {fmt(op)} = dot{n}({fmt(args[0])}, {fmt(args[1])});")
        elif op._op == "+":
            print(f"double {fmt(op)} = {' + '.join(fmt(v) for v in op.args())};")
        elif op._op == "array":
            n = len(op._prev)
            print(
                f"double {fmt(op)}[{n}] = {{ {', '.join(fmt(v) for v in op.args())} }};"
            )
        elif op._op == "":
            print(f"double {fmt(op)} = {op.data};")
        elif op._op == "input":
            print(f"double {fmt(op)} = in[{op.data}];")
        elif op._op == "ReLU":
            arg = fmt(op.arg(0))
            print(f"double {fmt(op)} = {arg} > 0 ? {arg} : 0;")
        else:
            raise RuntimeError(f"unexpected op {op._op!r}")


dim_in = 28 * 28
net = MLP(dim_in, [50, 10])
inp = hashcons_array(tuple(Value(i, (), "input") for i in range(dim_in)))
image = hashcons_array(inp._prev)
model = net(image._prev)
loss = hashcons_array(tuple(model))
# pretty(loss)
stderr = __import__("sys").stderr
before = len(loss.find().topo())
print(" ", count(loss.find()))
changed = optimize(loss.find())
after = len(loss.find().topo())
if changed:
    print(
        "before",
        before,
        "after",
        after,
        f"{(after-before)/before*100:.2f}%",
        file=stderr,
    )
    print(" ", OPT_LOG, file=stderr)
    print(" ", count(loss.find()), file=stderr)
# pretty(loss.find())
# compile(loss.find())
