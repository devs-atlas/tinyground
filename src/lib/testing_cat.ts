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

function sum_along_axis(t: NdA, axis: number){
  return;
}
// how to sum along axis when you only have access to full-array-sum
// [2,3,4]
function sum(t: NdA, axis?: number | number[]) {
  if (axis == undefined){
    return t.sum();
  }
  //do op if number
  if (typeof axis == "number"){
    return sum_along_axis(t, axis);
  } 

  let ans : NdA = t;
  //do op if number[]
  for (let i = t.shape.length - 1; i > -1; --i) {
    if(axis.includes(i)){
      ans = sum_along_axis(ans, i);
    }
  }
}

let a = nj.ones([2,3,4])
console.log(broadcast_to(a, [5,2,3,4]).shape);
