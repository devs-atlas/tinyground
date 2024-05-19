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
          console.log(result.data.data.arraySync());
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
        input: tf.tensor([1, 2, 3]),
        output: tf.tensor([[1, 2, 3], [1, 2, 3]]),
        args: [[2, 3]]
      }
    ]
  },
  {
    description: 'Tensor.transpose()',
    op: 'transpose',
    cases: [
      {
        input: tf.tensor([[1, 2, 3], [4, 5, 6]]),
        output: tf.tensor([[1, 4], [2, 5], [3, 6]]),
        args: [1, 0]
      }
    ]
  }
];

let x = new Tensor(tf.tensor([1, 2, 3]));
console.log('aa');
console.log(x.expand([1, 3]));
// testUnaryOps(unaryTests);
