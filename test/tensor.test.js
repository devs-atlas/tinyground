import * as tf from "@tensorflow/tfjs";
import Tensor from "../lib/tensor";

function close(x, y, epsilon = 0.001) {
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
  let data1 = [
    [1, 2],
    [3, 4],
    [5, 7],
  ];
  let t1 = new Tensor(data1);
  let data2 = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];
  let t2 = new Tensor(data2);

  test("add with tensor", () => {
    expect(t1.add(t2)).toEqual(
      data1.map((row, r) => row.map((x, c) => x + data2[r][c]))
    );
  });

  test("add with number", () => {
    expect(t1.add(5)).toEqual(data1.map((row) => row.map((x) => x + 5)));
  });

  test("sum", () => {
    expect(t1.sum()).toEqual([data1.flat(Infinity).reduce((a, b) => a + b)]);
  });

  test("max", () => {
    expect(t1.max()).toEqual([Math.max(...data1.flat(Infinity))]);
  });

  test("min", () => {
    expect(t1.min()).toEqual([Math.min(...data1.flat(Infinity))]);
  });

  test("tranpose with default axes", () => {
    expect(t1.transpose()).toEqual(
      data1[0].map((_, c) => data1.map((row) => row[c]))
    );
  });

  test("sqrt", () => {
    expect(t1.sqrt()).toEqual(data1.map((row) => row.map(Math.sqrt)));
  });

  test("relu", () => {
    expect(t1.relu()).toEqual(
      data1.map((row) => row.map((e) => (e > 0 ? e : 0)))
    );
  });

  // test("backward", () => {
  //   expect(t1.sum(undefined, true).backward().grad).toEqual([1, 1, 1, 1, 1]);
  // });
});
