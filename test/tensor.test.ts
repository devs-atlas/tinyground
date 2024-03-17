import * as tf from "@tensorflow/tfjs";
import Tensor from "../lib/tensor";

function close(x: tf.Tensor, y: tf.Tensor, epsilon = 0.001): boolean {
  const difference = x.sub(y).abs();
  return tf.max(difference).dataSync()[0] < epsilon;
}

expect.extend({
  toEqual(received: Tensor, expected: tf.Tensor) {
    if (close(received.data.data, expected)) {
      return {
        message: () => `tensors matched`,
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
    let t1 = new Tensor([
      [1, 2],
      [3, 4],
      [5, 7],
    ]);

    let t2 = new Tensor([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);

    const result = t1.add(t2);
    const expected = tf.tensor([
      [2, 4],
      [6, 8],
      [10, 13],
    ]);

    expect(result).toEqual(expected);
  });

  const data = [
    [3, 4],
    [5, 7],
    [1, 2],
  ];
  const tensor = new Tensor(data);

  test("sum", () => {
    expect(tensor.sum()).toEqual([data.flat().reduce((a, b) => a + b)]);
  });

  test("max", () => {
    expect(tensor.max()).toEqual([Math.max(...data.flat())]);
  });

  console.log(tensor.min().data.data.dataSync())

  test("min", () => {
    expect(tensor.min()).toEqual([Math.min(...data.flat())]);
  });
});
