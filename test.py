import random
from micrograd import nn
from micrograd.engine import Value

random.seed(1337)
n = nn.MLP(2, [16, 16, 1])
x = [Value(i) for i in range(16)]
print(n(x))
print("""\
#include <cstring>
#include <initializer_list>
#include <algorithm>

template <typename T = double, int dim = 1>
class Vector {
 public:
  Vector<T, dim>() {
    std::memset(arr, 0);
  }
  Vector<T, dim>(T other[dim]) {
    for (int i = 0; i < dim; i++) {
      arr[i] = other[i];
    }
  }
  Vector<T, dim>(std::initializer_list<T> other) {
    static_assert(other.size() == dim, "oh no");
    for (int i = 0; i < dim; i++) {
      arr[i] = other[i];
    }
  }
  Vector<T, dim> dot(Vector<T, dim> other) {
    T result[dim];
    for (int i = 0; i < dim; i++) {
      result[i] = arr[i] * other.arr[i];
    }
    return result;
  }
  T sum() {
    T result = 0;
    for (int i = 0; i < dim; i++) {
      result += arr[i];
    }
    return result;
  }
  T& at(int idx) { return arr[idx]; }

 private:
  T arr[dim];
};
""")
print("\n".join(n.compile(x)))
# y = n(x)
# print(y, n)
