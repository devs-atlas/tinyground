import * as tf from "@tensorflow/tfjs";
import Tensor from "../lib/tensor";

function close(x, y, epsilon = 0.001){
  const difference = x.sub(y).abs();
  return tf.max(difference).dataSync()[0] < epsilon;
}

expect.extend({
  toEqual(received, expected) {
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

    // @ts-ignore
    expect(tensor.sum()).toEqual([data.flat(Infinity).reduce((a, b) => a + b)]);
  });

  test("max", () => {
    // @ts-ignore
    expect(tensor.max()).toEqual([Math.max(...data.flat(Infinity))]);
  });

  test("min", () => {
    // @ts-ignore
    expect(tensor.min()).toEqual([Math.min(...data.flat(Infinity))]);
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

  test("sqrt", () => {
    let data1 = [4, 2, 5, 12];
    expect(new Tensor(data1).sqrt()).toEqual(data1.map((e) => Math.sqrt(e)));
  });

  test("relu", () => {
    let data1 = [-123, 4, 5, 3];
    expect(new Tensor(data1).relu()).toEqual(data1.map((e) => (e > 0 ? e : 0)));
  });

  test("backward", () => {
    let data1 = [1, 2, 3, 4, 5];

    let t3 = new Tensor(data1).sum(undefined, true).backward()
    expect(t3.grad).toEqual([1, 1, 1, 1, 1]);
  });
});
