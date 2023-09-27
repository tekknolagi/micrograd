import sys
from rpython.rlib import jit
import random
from mnist import MLP, Value

driver = jit.JitDriver(
    reds = 'auto',
    greens = ['model'],
)

random.seed(4)

def main(args):
    # crappy argument handling
    for i in range(len(args)):
        if args[i] == "--jit":
            if len(args) == i + 1:
                print "missing argument after --jit"
                return 2
            jitarg = args[i + 1]
            del args[i:i+2]
            jit.set_user_param(None, jitarg)
            break
    if len(args) >= 2:
        epochs = int(args[1])
        del args[1]
    else:
        epochs = 1000
    layers = 6
    if len(args) >= 2:
        layers = int(args[1])
        del args[1]
    model = MLP(2, [layers, 1])
    batch = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    for epoch in range(epochs):
        driver.jit_merge_point(model=model)

        model.zero_grad()
        epoch_loss = 0.
        jit.promote(len(batch))
        for xs, _ in batch:
            jit.promote(len(xs))
        outputs = [model.evalmlp([Value(x) for x in xs])[0] for xs, _ in batch]
        expected = [exp for _, exp in batch]
        loss = Value(0.0)
        for i in range(len(expected)):
            exp = expected[i]
            act = outputs[i]
            x = Value(exp).sub(act).pow(2)
            loss = loss.add(x)
        loss = loss.mul(Value(1.0 / len(expected)))
        loss.backward()
        epoch_loss += loss.data
        for p in model.parameters():
            p.data -= 0.1 * p.grad / len(batch)
        if epoch % 100 == 0:
            print "...epoch %s loss %s" % (epoch, epoch_loss/len(batch))


    print("params", [p.data for p in model.parameters()])
    for im in batch:
        result, = model.evalmlp([Value(x) for x in im[0]])
        assert abs(result.data-im[1]) < 0.1
    return 1

def target(*args):
    return main

if __name__ == '__main__':
    main(sys.argv)
