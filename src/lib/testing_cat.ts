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

  let ans = t;

  for (let i = out_shape.length - 1; i > -1; --i) {
    let times = Math.abs(out_shape[i] - shape[i]);
    if(times > 0){
        let stackedArray = Array.from({length: times+1}, () => ans);
        ans = nj.stack(stackedArray, i);
        let new_shape = ans.shape.filter((dim, idx) =>  idx != i+1)
        ans = ans.reshape(new_shape)
    }
  }

  return ans;
}

//3x1x2 -> 3x3x2

const a = nj.random([2,1,2]);
console.log(a)
console.log(broadcast_to(a, [2,2,2]));
