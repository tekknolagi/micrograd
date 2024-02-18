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
    return v._op == '' and v.data == val


def is_zero(v):
    return is_const(v, 0)


OPT_LOG = collections.Counter()


def optimize_one(v):
    if v._op == '+':
        args = v.args()
        if any(arg._op == '+' for arg in args):
            OPT_LOG['flatten_plus'] += 1
            new_args = []
            for arg in args:
                if arg._op == '+':
                    new_args.extend(arg.args())
                else:
                    new_args.append(arg)
            v.make_equal_to(Value(0, tuple(new_args), '+'))
            return True
        if any(is_zero(arg) for arg in args):
            OPT_LOG['plus_zero'] += 1
            v.make_equal_to(Value(0, filter(lambda arg: not is_const(arg, 0), args), '+'))
            return True
        if len(args) == 1:
            OPT_LOG['plus_single'] += 1
            v.make_equal_to(args[0])
            return True
    if v._op == '*':
        args = v.args()
        if any(is_zero(arg) for arg in args):
            OPT_LOG['mul_zero'] += 1
            v.make_equal_to(Value(0))
            return True
        if any(is_const(arg, 1) for arg in args):
            OPT_LOG['mul_one'] += 1
            v.make_equal_to(Value(0, filter(lambda arg: not is_const(arg, 1), args), '*'))
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
        if op._op == '+' and any(arg._op == '*' for arg in args):
            mul_args = tuple(arg for arg in args if arg._op == '*')
            mul_left = hashcons_array(tuple(arg.arg(0) for arg in mul_args))
            mul_right = hashcons_array(tuple(arg.arg(1) for arg in mul_args))
            other_args = tuple(arg for arg in args if arg._op != '*')
            op.make_equal_to(Value(0, (Dot(mul_left, mul_right), *other_args), '+'))
            changed = True
            continue
    return changed


def fmt(v):
    return f"v{v._id}"


def pretty(v):
    topo = v.topo()
    for op in topo:
        if op._op == 'input':
            print(f"{fmt(op)} = input")
        elif op._op == '':
            print(f"{fmt(op)} = {op.data}")
        else:
            print(f"{fmt(op)} = {op._op} {' '.join(fmt(c) for c in op.args())}")


def count(v):
    c = collections.Counter()
    for op in v.topo():
        c[op._op] += 1
    return c


dim_in = 28*28
net = MLP(dim_in, [50, 10])
model = net([Value(0, (), 'input') for _ in range(dim_in)])
loss = Array(model)
# pretty(loss)
changed = True
nrounds = 0
start = num_nodes(loss.find())
while changed:
    before = num_nodes(loss.find())
    changed = optimize(loss.find())
    after = num_nodes(loss.find())
    if changed:
        print("before", before, "after", after, f"{(after-before)/before*100:.2f}% (cum {(after-start)/start*100:.2f}%)")
        print(" ", OPT_LOG)
        print(" ", count(loss.find()))
    OPT_LOG.clear()
    nrounds += 1
# pretty(loss.find())
