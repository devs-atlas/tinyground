import { default as nj } from "@d4c/numjs";
import { Tensor } from "./teeny";

expect.extend({
  toEqual(received: Tensor, expected: Tensor) {
    const pass = received.data.toJSON() === expected.data.toJSON();
    if (pass) {
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

describe("nick is erect", () => {
  test("should cover him", () => {
    let t1 = new Tensor(
      nj.array([
        [1, 2],
        [3, 4],
        [5, 7],
      ]),
      false
    );

    let t2 = new Tensor(
      nj.array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]),
      true
    );

    const result = t1.add(t2);
    const expected = new Tensor(
      nj.array([
        [2, 4],
        [6, 8],
        [10, 13],
      ]),
      false
    );

    expect(result).toEqual(expected);
  });
});
