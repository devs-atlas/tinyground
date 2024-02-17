import { NdArray as NdA, default as nj } from "@d4c/numjs";

// should you be able to broadcast_to a [2,3] to [2,4,3]?
// i dont think so but wanna be sure and make the error handling for that clearer
// maybe do an is_broadcastable function or smth
// maybe that's handled in Tensor.reduce?

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

  let ans = t;

  for (let i = out_shape.length - 1; i > -1; --i) {
    let times = Math.abs(out_shape[i] - shape[i]);
    if (times > 0) {
      let stackedArray = Array.from({ length: times + 1 }, () => ans);
      ans = nj.stack(stackedArray, i);
    }
  }

  return ans;
}

function sum_along_axis(t: NdA, axis: number): NdA {
  let slices = Array(t.shape[axis])
    .fill(null)
    .map((_, i) => {
      return t.shape.map((dim, j) => {
        if (j === axis) {
          return [i, i + 1];
        }
        return [0, dim];
      });
    });

  let ans = nj.zeros(t.slice(...slices[0]).shape);

  for (let slice of slices) {
    ans.add(t.slice(...slice), false);
  }

  return ans;
}

function sum(t: NdA, axis?: number | number[]) {
  if (axis === undefined) {
    return nj.array(t.sum());
  }
  if (typeof axis === "number"){
    return sum_along_axis(t, axis)
  }
  let ans = t;
  for(let i=0; i < t.shape.length; i++){
    if(axis.includes(i)){
      ans = sum_along_axis(ans, i); 
    }      
  }

  return ans;
}

