import { NdArray as NdA, default as nj } from "@d4c/numjs";

function broadcast_to(t: NdA, shape: number[]) {
  if (t.shape.length > shape.length) {
    throw Error(`Cannot broadcast shape ${t.shape} to smaller shape ${shape}.`);
  }

  let out_shape = Array(shape.length - t.shape.length)
    .fill(1)
    .concat(t.shape);

  for (let i = t.shape.length - 1; i > -1; --i) {
    if (out_shape[i] !== shape[i] && out_shape[i] !== 1)
      throw Error("Mismatched broadcast dimensions");
  }

  const ans = t;

  for (let i = out_shape.length - 1; i > -1; --i) {
    let times = out_shape[i] - shape[i];
    while (times--) {
      // ans.set(...out_shape.slice([-i]), t.slice(...out_shape.slice([-i])));
      ans.concat_along_axis(i, ans)
    }
  }

  return ans;
}

// 2x1x3 -> 2x4x3

const a = nj.random([3]);

console.log(`Before reshape: ${a}`);

// const b = broadcastTo(a, [3, 3])

console.log(`broadcast: ${nj.broadcast([5, 1, 3, 4], [5, 3, 3, 4])}`);

const original_shape = [1, 3, 4];
const from = nj
  .arange(original_shape.reduce((p, n) => p * n, 1))
  .reshape(...original_shape);
console.log(`before: ${from}`);
const new_shape = [4, 4, 4];
// console.log(broadcast_to(from, new_shape));
// console.log(`${broadcast_to(nj.array([1, 4, 4]), [2, 4, 1])}`);

// console.log(`After reshape: ${b}`);
