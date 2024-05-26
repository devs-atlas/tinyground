//TODO: Do gradient testing in each op too
import { describe, expect, jest, test } from "@jest/globals";
import Tensor from "../lib/tensor.js";
import * as tf from "@tensorflow/tfjs";

/*
 * format of a case:
 * {
 * input: tf.tensor,
 * output: tf.tensor,
 * args: [] - the length of args must equal number of inputs to args
 * }
 */

// SETUP

function isNumClose(num1, num2, tolerance) {
  return Math.abs(num1 - num2) <= tolerance;
}

function areTensorsClose(tensor1, tensor2, tolerance = 1e-10) {
  if (tensor1.length !== tensor2.length) {
    return false;
  }
  for (let i = 0; i < tensor1.length; i++) {
    if (Array.isArray(tensor1[i]) && Array.isArray(tensor2[i])) {
      if (!areTensorsClose(tensor1[i], tensor2[i], tolerance)) {
        return false;
      }
    } else if (!isNumClose(tensor1[i], tensor2[i], tolerance)) {
      return false;
    }
  }
  return true;
}

function testReduceOps(tests) {
  tests.forEach(testOp => {
    describe(testOp.description, () => {
      testOp.cases.forEach((c, i) => {
        test(`case #${i + 1}`, () => {

          const tensor = new Tensor(c["input"]);
          const result = tensor[testOp.op](...c["args"]);
          console.log(`Input Tensor: ${tensor.data.data.arraySync()}`);
          console.log(`Output Tensor: ${result.data.data.arraySync()}`);
          expect(result.data.data.arraySync()).toBeApproximatelyEqual(
            c["output"].arraySync()
          );
        });
      })
    });
  });
}

function testUnaryOps(tests) {
  tests.forEach(testOp => {
    describe(testOp.description, () => {
      testOp.cases.forEach((c, i) => {
        test(`case #${i + 1}`, () => {
          const tensor = new Tensor(c["input"]);
          const result = tensor[testOp.op](...c["args"]);
          expect(result.shape).toEqual(c["output"].shape);
          expect(result.data.data.arraySync()).toBeApproximatelyEqual(
            c["output"].arraySync()
          );
        });
      });
    });
  })
}

function testLoadOps(tests) {
  tests.forEach(testOp => {
    describe(testOp.description, () => {
      testOp.cases.forEach((c, i) => {
        test(`case #${i + 1}`, () => {
          const result = Tensor[testOp.op](...c["args"]);
          expect(result.shape).toEqual(c.output.shape);
          expect(result.data.data.arraySync()).toBeApproximatelyEqual(
            c["output"].arraySync(),
          );
        });
      });
    });
  })
}

function testBinaryOps(tests) {
  tests.forEach(testOp => {
    describe(testOp.description, () => {
      testOp.cases.forEach((c, i) => {
        test(`case #${i + 1}`, () => {
          const tensor1 = new Tensor(c.input[0]);
          const tensor2 = new Tensor(c.input[1]);
          //NOTE: No binary ops have
          const result = tensor1[testOp.op](tensor2, ...c["args"]);
          expect(result.shape).toEqual(c.output.shape);
          expect(result.data.data.arraySync()).toBeApproximatelyEqual(
            c.output.arraySync(),
          );
        });
      });
    });
  });
}

expect.extend({
  toBeApproximatelyEqual(array1, array2, precision = 1e-6) {
    const compareArrays = (arr1, arr2, tol) => {
      if (arr1.length !== arr2.length) return false;
      for (let i = 0; i < arr1.length; i++) {
        if (Array.isArray(arr1[i]) && Array.isArray(arr2[i])) {
          if (!compareArrays(arr1[i], arr2[i], tol)) return false;
        } else {
          if (typeof arr1[i] === "number" && typeof arr2[i] === "number") {
            if (Math.abs(arr1[i] - arr2[i]) > tol) return false;
          } else {
            if (arr1[i] !== arr2[i]) return false;
          }
        }
      }
      return true;
    };

    const pass = compareArrays(array1, array2, precision);
    return {
      message: () =>
        pass
          ? `expected arrays not to be approximately equal within a tolerance of ${precision}, but they were`
          : `expected arrays to be approximately equal within a tolerance of ${precision}, but they were not`,
      pass: pass,
    };
  },
});

// TESTS

const unaryTests = [
  {
    description: 'Tensor.neg()',
    op: 'neg',
    cases: [
      {
        input: tf.tensor([1, -2, 3]),
        output: tf.tensor([-1, 2, -3]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.log()',
    op: 'log',
    cases: [
      {
        input: tf.tensor([1, Math.E, Math.E ** 2]),
        output: tf.tensor([0, 1, 2]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.log2()',
    op: 'log2',
    cases: [
      {
        input: tf.tensor([1, 2, 4, 8]),
        output: tf.tensor([0, 1, 2, 3]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.exp()',
    op: 'exp',
    cases: [
      {
        input: tf.tensor([0, 1, 2]),
        output: tf.tensor([1, Math.E, Math.E ** 2]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.exp2()',
    op: 'exp2',
    cases: [
      {
        input: tf.tensor([0, 1, 2, 3]),
        output: tf.tensor([1, 2, 4, 8]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.relu()',
    op: 'relu',
    cases: [
      {
        input: tf.tensor([-1, 0, 1, -2, 2]),
        output: tf.tensor([0, 0, 1, 0, 2]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.sigmoid()',
    op: 'sigmoid',
    cases: [
      {
        input: tf.tensor([-1, 0, 1]),
        output: tf.tensor([1 / (1 + Math.exp(1)), 0.5, 1 / (1 + Math.exp(-1))]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.sqrt()',
    op: 'sqrt',
    cases: [
      {
        input: tf.tensor([0, 1, 4, 9]),
        output: tf.tensor([0, 1, 2, 3]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.rsqrt()',
    op: 'rsqrt',
    cases: [
      {
        input: tf.tensor([1, 4, 9]),
        output: tf.tensor([1, 0.5, 1 / 3]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.sin()',
    op: 'sin',
    cases: [
      {
        input: tf.tensor([0, Math.PI / 2, Math.PI]),
        output: tf.tensor([0, 1, 0]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.cos()',
    op: 'cos',
    cases: [
      {
        input: tf.tensor([0, Math.PI / 2, Math.PI]),
        output: tf.tensor([1, 0, -1]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.tan()',
    op: 'tan',
    cases: [
      {
        input: tf.tensor([0, Math.PI / 4, Math.PI / 2]),
        output: tf.tensor([0, 1, Infinity]),
        args: []
      }
    ]
  },
  {
    description: 'Tensor.permute()',
    op: 'permute',
    cases: [
      {
        input: tf.tensor([[1, 2, 3], [4, 5, 6]]),
        output: tf.tensor([[1, 4], [2, 5], [3, 6]]),
        args: [[1, 0]]
      },
      {
        input: tf.ones([8, 3, 2]),
        output: tf.ones([8, 2, 3]),
        args: [[0, 2, 1]]
      }
    ]
  },
  {
    description: 'Tensor.reshape()',
    op: 'reshape',
    cases: [
      {
        input: tf.tensor([1, 2, 3, 4]),
        output: tf.tensor([[1, 2], [3, 4]]),
        args: [[2, 2]]
      }
    ]
  },
  {
    description: 'Tensor.expand()',
    op: 'expand',
    cases: [
      {
        input: tf.ones([2, 1, 4]),
        output: tf.ones([2, 3, 4]),
        args: [[-1, 3, -1]]
      },
      {
        input: tf.ones([2, 1, 4]),
        output: tf.ones([2, 3, 4]),
        args: [[2, 3, 4]]
      },
    ]
  },
  {
    description: 'Tensor.transpose()',
    op: 'transpose',
    cases: [
      {
        input: tf.ones([3, 1, 2]),
        output: tf.ones([3, 2, 1]),
        args: [1, 2]
      },
      {
        input: tf.ones([3, 1]),
        output: tf.ones([1, 3]),
        args: [0, 1]
      }
    ]
  },
];

const reduceTests = [
  {
    description: 'Tensor.sum()',
    op: 'sum',
    cases: [
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor(10),
        args: []
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([4, 6]),
        args: [0]
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([3, 7]),
        args: [1]
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([[4, 6]]),
        args: [0, true]
      }
    ]
  },
  {
    description: 'Tensor.max()',
    op: 'max',
    cases: [
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor(4),
        args: []
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([3, 4]),
        args: [0]
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([2, 4]),
        args: [1]
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([[3, 4]]),
        args: [0, true]
      }
    ]
  },
  {
    description: 'Tensor.min()',
    op: 'min',
    cases: [
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor(1),
        args: []
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([1, 2]),
        args: [0]
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([1, 3]),
        args: [1]
      },
      {
        input: tf.tensor([[1, 2], [3, 4]]),
        output: tf.tensor([[1, 2]]),
        args: [0, true]
      }
    ]
  },
];

const binaryTests = [
  {
    description: 'Tensor.add()',
    op: 'add',
    cases: [
      {
        input: [tf.tensor([1, 2, 3]), tf.tensor([1, 1, 1])],
        output: tf.tensor([2, 3, 4]),
        args: []
      },
      {
        input: [tf.tensor([1, 2, 3]), 1],  // Broadcasting a scalar
        output: tf.tensor([2, 3, 4]),
        args: []
      },
      {
        input: [tf.tensor([[1, 2], [3, 4]]), tf.tensor([[1], [1]])],  // Broadcasting a column vector
        output: tf.tensor([[2, 3], [4, 5]]),
        args: []
      },
    ]
  },
  {
    description: 'Tensor.sub()',
    op: 'sub',
    cases: [
      {
        input: [tf.tensor([3, 4, 5]), tf.tensor([1, 2, 3])],
        output: tf.tensor([2, 2, 2]),
        args: []
      },
      {
        input: [tf.tensor([3, 4, 5]), 1],  // Broadcasting a scalar
        output: tf.tensor([2, 3, 4]),
        args: []
      },
      {
        input: [tf.tensor([[3, 4], [5, 6]]), tf.tensor([[1], [1]])],  // Broadcasting a column vector
        output: tf.tensor([[2, 3], [4, 5]]),
        args: []
      },
    ]
  },
  {
    description: 'Tensor.mul()',
    op: 'mul',
    cases: [
      {
        input: [tf.tensor([1, 2, 3]), tf.tensor([2, 2, 2])],
        output: tf.tensor([2, 4, 6]),
        args: []
      },
      {
        input: [tf.tensor([1, 2, 3]), 2],  // Broadcasting a scalar
        output: tf.tensor([2, 4, 6]),
        args: []
      },
      {
        input: [tf.tensor([[1, 2], [3, 4]]), tf.tensor([[2], [2]])],  // Broadcasting a column vector
        output: tf.tensor([[2, 4], [6, 8]]),
        args: []
      },
    ]
  },
  {
    description: 'Tensor.div()',
    op: 'div',
    cases: [
      {
        input: [tf.tensor([4, 6, 8]), tf.tensor([2, 2, 2])],
        output: tf.tensor([2, 3, 4]),
        args: []
      },
      {
        input: [tf.tensor([4, 6, 8]), 2],  // Broadcasting a scalar
        output: tf.tensor([2, 3, 4]),
        args: []
      },
      {
        input: [tf.tensor([[4, 6], [8, 10]]), tf.tensor([[2], [2]])],  // Broadcasting a column vector
        output: tf.tensor([[2, 3], [4, 5]]),
        args: []
      },
    ]
  },
  {
    description: 'Tensor.dot()',
    op: 'dot',
    cases: [
      {
        input: [tf.ones([3, 2]), tf.ones([2, 3])],
        output: tf.fill([3, 3], 2),
        args: []
      },
      {
        input: [tf.ones([3, 2]), tf.ones([2])],
        output: tf.fill([3], 2),
        args: []
      },
      {
        input: [tf.ones([3]), tf.ones([3, 2])],
        output: tf.fill([2], 3),
        args: []
      },
      {
        input: [tf.ones([3]), tf.ones([3])],
        output: tf.tensor(3),
        args: []
      },
    ],
  }
];

const loadOpTests = [
  {
    description: 'Tensor.ones()',
    op: 'ones',
    cases: [
      {
        args: [[2, 2], false],  // shape and requires_grad
        output: tf.tensor([[1, 1], [1, 1]]),
      },
      {
        args: [[3], true],  // testing with requires_grad
        output: tf.tensor([1, 1, 1]),
      }
    ]
  },
  {
    description: 'Tensor.zeros()',
    op: 'zeros',
    cases: [
      {
        args: [[2, 3], false],
        output: tf.tensor([[0, 0, 0], [0, 0, 0]]),
      },
      {
        args: [[1, 4], true],
        output: tf.tensor([[0, 0, 0, 0]]),
      }
    ]
  },
  {
    description: 'Tensor.full()',
    op: 'full',
    cases: [
      {
        args: [[2, 2], 7, false],  // shape, fill_value, requires_grad
        output: tf.tensor([[7, 7], [7, 7]]),
      },
      {
        args: [[1, 3], 3, true],
        output: tf.tensor([[3, 3, 3]]),
      }
    ]
  }
];

testUnaryOps(unaryTests);
testBinaryOps(binaryTests);
testLoadOps(loadOpTests);
testReduceOps(reduceTests);
