import * as tf from "@tensorflow/tfjs";
import Tensor from "../lib/tensor";

function close(x, y, epsilon = 0.001) {
  const difference = x.sub(y).abs();
  return tf.max(difference).dataSync()[0] < epsilon;
}

function arrayEquals(a, b) {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; ++i) if (a[i] != b[i]) return false;
  return true;
}

expect.extend({
  toEqual(received, expected) {
    let t = new Tensor(expected);

    const shapeMsg = `Got shape ${received.shape}; expected shape ${t.shape}`;

    if (!arrayEquals(t.shape, received.shape)) {
      return {
        message: () => shapeMsg,
        pass: false,
      };
    }
    if (close(received.data.data, expected)) {
      return {
        message: () => `tensors matched; ${shapeMsg}`,
        pass: true,
      };
    } else {
      return {
        message: () => `expected tensors to be equal; ${shapeMsg}`,
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
  let t1 = new Tensor(data1, true);
  let data2 = [
    [1, 2],
    [3, 4],
    [5, 6],
  ];
  let t2 = new Tensor(data2, true);

  test("add with tensor", () => {
    expect(t1.add(t2)).toEqual(
      data1.map((row, r) => row.map((x, c) => x + data2[r][c]))
    );
  });

  test("add with number", () => {
    expect(t1.add(5)).toEqual(data1.map((row) => row.map((x) => x + 5)));
  });

  test("sum", () => {
    expect(t1.sum(undefined, true)).toEqual([
      [data1.flat(Infinity).reduce((a, b) => a + b)],
    ]);
  });

  test("max", () => {
    expect(t1.max(undefined, true)).toEqual([
      [Math.max(...data1.flat(Infinity))],
    ]);
  });

  test("min", () => {
    expect(t1.min(undefined, true)).toEqual([
      [Math.min(...data1.flat(Infinity))],
    ]);
  });

  test("tranpose with default axes", () => {
    expect(t1.transpose()).toEqual(
      data1[0].map((_, c) => data1.map((row) => row[c]))
    );
  });

  test("sqrt", () => {
    expect(t1.sqrt()).toEqual(data1.map((row) => row.map(Math.sqrt)));
  });
  test("transpose", () => {
    const expected = [[1, 3, 5], [2, 4, 7]];
    expect(t1.transpose()).toEqual(expected);
  });
  test("dot", () => {
    const expected = [[5, 11, 17], [11, 25, 39], [19, 43, 67]]
    let t2T = t2.transpose();
    let out = t1.dot(t2T);
    expect(out).toEqual(expected);
  })

  test("relu", () => {
    expect(t1.relu()).toEqual(
      data1.map((row) => row.map((e) => (e > 0 ? e : 0)))
    );
  });

  test("backward", () => {
    let out = t1.add(t2).sum(undefined, true);
    out = out.backward()
    expect(t1.grad).toEqual([[1, 1], [1, 1], [1, 1]]);
  });
});
