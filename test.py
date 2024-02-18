import collections
from micrograd.engine import Value, Dot, Array
from micrograd.nn import MLP


def num_nodes(val):
    todo = [val]
    visited = set()
    while todo:
        v = todo.pop()
        if v in visited:
            continue
        visited.add(v)
        args = v.args()
        todo.extend(args)
    return len(visited)


def is_const(v, val):
    return v._op == "" and v.data == val


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
        if any(is_const(arg, 0) for arg in args):
            OPT_LOG["plus_zero"] += 1
            non_zero = tuple(filter(lambda arg: not is_const(arg, 0), args))
            v.make_equal_to(Value(0, non_zero, "+"))
            return True
        if len(args) == 1:
            OPT_LOG["plus_single"] += 1
            v.make_equal_to(args[0])
            return True
    return False


def optimize(v):
    topo = v.topo()
    changed = False
    for op in topo:
        changed |= optimize_one(op.find())
    topo = v.find().topo()

    array_seen = {}

    def hashcons_array(vs):
        if vs not in array_seen:
            array_seen[vs] = Array(vs)
        return array_seen[vs]

    for op in reversed(topo):
        args = op.args()
        if op._op == "+" and any(arg._op == "*" for arg in args):
            mul_args = tuple(arg for arg in args if arg._op == "*")
            mul_left = hashcons_array(tuple(arg.arg(0) for arg in mul_args))
            mul_right = hashcons_array(tuple(arg.arg(1) for arg in mul_args))
            other_args = tuple(arg for arg in args if arg._op != "*")
            op.make_equal_to(Value(0, (Dot(mul_left, mul_right), *other_args), "+"))
            changed = True
            continue
    return changed


def fmt(v):
    if v._op == "":
        return str(v.data)
    return f"v{v._id}"


def pretty(v):
    topo = v.topo()
    for op in topo:
        if op._op == "input":
            print(f"{fmt(op)} = input")
        elif op._op == "":
            pass
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
        elif op._op == "array":
            n = len(op._prev)
            print(
                f"double {fmt(op)}[{n}] = {{ {', '.join(fmt(v) for v in op.args())} }};"
            )
        elif op._op == "":
            pass
        elif op._op == "input":
            print(f"double {fmt(op)} = in[{op.data}];")
        elif op._op == "ReLU":
            arg = fmt(op.arg(0))
            print(f"double {fmt(op)} = {arg} > 0 ? {arg} : 0;")
        else:
            raise RuntimeError(f"unexpected op {op._op!r}")


dim_in = 28 * 28
net = MLP(dim_in, [50, 10])
model = net([Value(i, (), "input") for i in range(dim_in)])
loss = Array(model)
# pretty(loss)
changed = True
nrounds = 0
start = num_nodes(loss.find())
stderr = __import__("sys").stderr
while changed:
    before = num_nodes(loss.find())
    changed = optimize(loss.find())
    after = num_nodes(loss.find())
    if changed:
        print(
            "before",
            before,
            "after",
            after,
            f"{(after-before)/before*100:.2f}% (cum {(after-start)/start*100:.2f}%)",
            file=stderr,
        )
        print(" ", OPT_LOG, file=stderr)
        print(" ", count(loss.find()), file=stderr)
    OPT_LOG.clear()
    nrounds += 1
# pretty(loss.find())
# compile(loss.find())
