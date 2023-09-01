import micrograd
import random
from micrograd.engine import Value
from micrograd.nn import MLP

random.seed(1337)

# model = MLP(2, [2, 1])
# inp = [Value(123), Value(456)]
# out = model(inp)
# topo = out.topo()
# for o in topo:
#     print("\n".join(o.compile()))

# dim = 784
# model = MLP(dim, [512, 10])
dim = 20
model = MLP(dim, [6, 10])
inp = [Value(0, (), 'input') for _ in range(dim)]
out = model(inp)
expected_onehot = [Value(0) for _ in range(10)]
expected_onehot[3] = Value(1)
loss = sum((exp-act)**2 for exp,act in zip(expected_onehot, out))
topo = loss.topo()
print("#include <math.h>")
print("#include <stdio.h>")
print(f"double data[{micrograd.engine.counter}];")
print(f"double grad[{micrograd.engine.counter}];")
print("double relu(double x) { if (x < 0) { return 0; } else { return x; } }")
print("void init() {")
for o in model.parameters():
    print(f"data[{o._id}] = {o.data}L;")
print("}")
print("void input() {")
for o in inp:
    # TODO(max): Read image and also update label in loss
    print(f"data[{o._id}] = {o.data}L;")
print("}")
print("void forward() {")
for o in topo:
    lines = o.compile()
    if lines:
        print("\n".join(lines))
print("}")
print("void zero_grad() {")
for o in model.parameters():
    print(f"grad[{o._id}] = 0;")
print("}")
print("void backward() {")
print(f"grad[{loss._id}] = 1;")
for o in reversed(topo):
    lines = o.backward_compile()
    if lines:
        print("\n".join(lines))
print("}")
print("void update(int step) {")
# TODO(max): It's not always 100; is this hard-coded for number of training
# rounds in Karpathy's code?
print("double learning_rate = 1.0L - (0.9L * (double)step) / 100.0L;")
for o in model.parameters():
    print(f"data[{o._id}] -= learning_rate * grad[{o._id}];")
print("}")
print("""
#define NUM_LOSSES 16
#define EPS 1
double losses[NUM_LOSSES];
int losses_idx = 0;
void add_loss(double loss) {
  losses[losses_idx] = loss;
  losses_idx = (losses_idx + 1) % NUM_LOSSES;
}
double diff(double x, double y) {
  if (x > y) return x - y;
  return y - x;
}
int loss_changing() {
  double l = losses[losses_idx];
  int count = 0;
  for (int i = losses_idx, j = 0; j < NUM_LOSSES; j++) {
    if (isnan(losses[i]) || isinf(losses[i])) {
      // Something went wrong; stop training.
      return 0;
    }
    if (diff(losses[i], l) < EPS) {
      count++;
    }
    i = (i + 1) % NUM_LOSSES;
  }
  return count < NUM_LOSSES;
}
""")
print(f"""int main() {{
  init();
  int nrounds = 0;
  add_loss(1);
  while (loss_changing()) {{
      // TODO(max): Batches
      input();
      forward();
      add_loss(data[{loss._id}]);
      zero_grad();
      backward();
      update(nrounds);
      printf("round %d: loss is %lf\\n", nrounds, data[{loss._id}]);
      nrounds++;
  }}
  printf("%d training rounds\\n", nrounds);
""")
for idx, i in enumerate(out):
    print(f"""printf("digit {idx}: %lf\\n", data[{i._id}]);""")
print("}")
