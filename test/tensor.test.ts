import * as tf from "@tensorflow/tfjs";
import Tensor from "../lib/tensor";

function close(x: tf.Tensor, y: tf.TensorLike, epsilon = 0.001): boolean {
  const difference = x.sub(y).abs();
  return tf.max(difference).dataSync()[0] < epsilon;
}

expect.extend({
  toEqual(received: Tensor, expected) {
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
  test("add with tensor", () => {
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

    expect(result).toEqual([
      [2, 4],
      [6, 8],
      [10, 13],
    ]);
  });

  test("add with number", () => {
    let t1 = new Tensor([
      [1, 2],
      [3, 4],
      [5, 7],
    ]);

    expect(t1.add(5)).toEqual([
      [6, 7],
      [8, 9],
      [10, 12],
    ]);
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

  test("min", () => {
    expect(tensor.min()).toEqual([Math.min(...data.flat())]);
  });

  test("tranpose with default axes", () => {
    let t1 = new Tensor([
      [1, 2],
      [3, 4],
      [5, 7],
    ]);

    expect(t1.transpose()).toEqual([
      [1, 3, 5],
      [2, 4, 7],
    ]);
  });
});
