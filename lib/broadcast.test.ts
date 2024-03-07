import * as tf from "@tensorflow/tfjs";
import { Tensor } from "./teeny";

expect.extend({
  toEqual(received: Tensor, expected: Tensor) {
    const equal = received.data.equal(expected.data).sum().dataSync()[0];
    const size = received.data.shape.reduce((x, y) => (x *= y));

    if (equal === size) {
      return {
        message: () => `expected tensors not to be equal`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected tensors to be equal`,
        pass: false,
      };
    }
  },
});

describe("Basic Tensor Ops", () => {
  test("add", () => {
    let t1 = new Tensor(
      tf.tensor([
        [1, 2],
        [3, 4],
        [5, 7],
      ]),
      false
    );

    let t2 = new Tensor(
      tf.tensor([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
      true
    );

    const result = t1.add(t2);
    const expected = new Tensor(
      tf.tensor([
        [2, 4],
        [6, 8],
        [10, 13],
      ]),
      false
    );

    expect(result).toEqual(expected);
  });
});
