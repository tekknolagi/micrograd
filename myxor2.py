from rpython.rlib import jit
import random
from mnist import MLP, Value

driver = jit.JitDriver(
    reds = 'auto',
    greens = ['model'],
)

random.seed(4)

def main(args):
    model = MLP(2, [6, 1])
    batch = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
    for epoch in range(1000):
        driver.jit_merge_point(model=model)

        model.zero_grad()
        epoch_loss = 0.
        jit.promote(len(batch))
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
    main([])
