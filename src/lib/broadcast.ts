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
  }

  return ans;
}

// [2,3,5]
// [2,3,4,5]
