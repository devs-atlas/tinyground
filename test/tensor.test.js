//TODO: Do gradient testing in each op too
import { describe, expect, jest, test } from '@jest/globals';
import Tensor from "../lib/tensor.js";
import * as tf from '@tensorflow/tfjs';

/*
  * format of a case:
  * {
  * input: tf.tensor,
  * output: tf.tensor,
  * args: []
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

function testUnaryOp(description, op, cases) {
  describe(description, () => {
    cases.forEach((c, i) => {
      test("case #${i+1}", () => {
        const tensor = new Tensor(c['input']);
        const result = tensor[op]();
        expect(result.shape).toEqual(c['output'].shape);
        expect(result.data.data.arraySync()).toBeApproximatelyEqual(c['output'].arraySync());
      });
    });
  });
}

function testLoadOp(description, opName, cases) {
  describe(description, () => {
    cases.forEach((c, i) => {
      test(`case #${i + 1}`, () => {
        const result = Tensor[opName](...c['input']);
        expect(result.shape).toEqual(c.output.shape);
        expect(result.data.data.arraySync()).toBeApproximatelyEqual(c['output'].data.data.arraySync());
      });
    });
  });
}

function testBinaryOp(description, opName, cases) {
  describe(description, () => {
    cases.forEach((c, i) => {
      test(`case #${i + 1}`, () => {
        const tensor1 = new Tensor(c.input[0]);
        const tensor2 = new Tensor(c.input[1]);
        const result = tensor1[opName](tensor2);
        expect(result.shape).toEqual(c.output.shape);
        expect(result.data.data.arraySync()).toBeApproximatelyEqual(c.output.arraySync());
      });
    });
  });
}

expect.extend({
  toBeApproximatelyEqual(array1, array2, precision = 1e-10) {
    const compareArrays = (arr1, arr2, tol) => {
      if (arr1.length !== arr2.length) return false;
      for (let i = 0; i < arr1.length; i++) {
        if (Array.isArray(arr1[i]) && Array.isArray(arr2[i])) {
          if (!compareArrays(arr1[i], arr2[i], tol)) return false;
        } else {
          if (typeof arr1[i] === 'number' && typeof arr2[i] === 'number') {
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
        pass ?
          `expected arrays not to be approximately equal within a tolerance of ${precision}, but they were` :
          `expected arrays to be approximately equal within a tolerance of ${precision}, but they were not`,
      pass: pass
    };
  }
});

// TESTS
