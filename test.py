import collections
import sys
from micrograd.engine import Value, Dot
from micrograd.nn import MLP


sys.setrecursionlimit(100_000)


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
    return len(tuple(v for v in visited if v._op != ''))


def optimize_one(v):
    if v._op == '+':
        args = []
        found = False
        for arg in v.args():
            if arg._op == '+':
                args += arg.args()
                found = True
            else:
                args.append(arg)
        if found:
            v.make_equal_to(Value(0, tuple(args), '+'))
            return True
        # left, *right = v.args()
        # if left._op == '+':
        #     v.make_equal_to(Value(0, (*left.args(), *right), '+'))
        #     return True
        # if right._op == '+':
        #     v.make_equal_to(Value(0, (left, *right.args()), '+'))
        #     return True
    return False


def optimize(v):
    topo = v.topo()
    changed = False
    for op in topo:
        changed |= optimize_one(op.find())
    topo = v.find().topo()
    for op in reversed(topo):
        args = op.args()
        if op._op == '+' and any(arg._op == '*' for arg in args):
            mul_args = tuple(arg for arg in args if arg._op == '*')
            mul_left = tuple(arg.arg(0) for arg in mul_args)
            mul_right = tuple(arg.arg(1) for arg in mul_args)
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
        if op._op == '':
            print(f"{fmt(op)} = input")
        elif op._op == 'dot':
            print(f"{fmt(op)} = dot [{', '.join(fmt(c.find()) for c in op._left)}] [{', '.join(fmt(c.find()) for c in op._right)}]")
        else:
            print(f"{fmt(op)} = {op._op} {' '.join(fmt(c) for c in op.args())}")


dim_in = 28*28
net = MLP(dim_in, [50, 10])
model = net([0]*dim_in)
loss = sum(model)
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
    nrounds += 1
# pretty(loss.find())
c = collections.Counter()
for op in loss.find().topo():
    c[op._op] += 1
print(c)
