import { NdArray as NdA, default as nj } from "@d4c/numjs";
import ops from "ndarray-ops";
import { BinaryOps, LoadOps, UnaryOps } from "./ops";
import { broadcast_to } from "./teeny";

// @ts-ignore
NdA.prototype.emax = function (x, copy = true) {
  if (arguments.length === 1) {
    copy = true;
  }
  const arr = copy ? this.clone() : this;
  // @ts-ignore
  x = NdA.new(x, this.dtype);
  ops.maxeq(arr.selection, x.selection);
  return arr;
};

// @ts-ignore
NdA.prototype.lt = function (x, copy = true) {
  if (arguments.length === 1) {
    copy = true;
  }
  const arr = copy ? this.clone() : this;
  // @ts-ignore
  x = NdA.new(x, this.dtype);
  ops.lteq(arr.selection, x.selection);
  return arr;
};

class LazyBuffer {
  data: NdA;

  constructor(buf: NdA) {
    this.data = buf;
  }

  get realized(): NdA {
    return this.data;
  }

  get shape(): number[] {
    return this.data.shape;
  }

  static loadop(op: LoadOps, shape: number[], arg?: number): LazyBuffer {
    switch (op) {
      case "RAND":
        return new LazyBuffer(nj.random(shape));
      case "CONST":
        return new LazyBuffer(nj.ones(shape).multiply(arg || 1));
      case "EMPTY":
        return new LazyBuffer(nj.empty(shape));
      default:
        throw new Error("Can only load RAND, CONST, or EMPTY Loadop.");
    }
  }

  e(op: UnaryOps | BinaryOps, ...srcs: LazyBuffer[]): LazyBuffer {
    let out = this.data;
    switch (op) {
      case "NEG":
        out = nj.negative(out);
        break;
      case "EXP2":
        out = nj.power(out, 2);
        break;
      case "LOG2":
        out = nj.log(out).divide(nj.log(2));
        break;
      case "SIN":
        out = nj.sin(out);
        break;
      case "SQRT":
        out = nj.sqrt(out);
        break;
      case "ADD":
        out = nj.add(this.data, srcs[0].data);
        break;
      case "SUB":
        out = nj.subtract(this.data, srcs[0].data);
        break;
      case "MUL":
        out = nj.multiply(this.data, srcs[0].data);
        break;
      case "DIV":
        out = nj.divide(this.data, srcs[0].data);
        break;
      case "MAX":
        if (this.shape !== srcs[0].shape) {
          const thisSize = this.shape.reduce((acc, e) => acc * e, 1);
          const srcsSize = srcs[0].shape.reduce((acc, e) => acc * e, 1);
          if (thisSize > srcsSize) {
            srcs[0].data = broadcast_to(srcs[0].data, this.shape);
          } else if (thisSize < srcsSize) {
            this.data = broadcast_to(this.data, srcs[0].shape);
          }
        }
        // @ts-ignore
        out = this.data.emax(srcs[0].data);
        break;
      case "CMPLT":
        if (this.shape !== srcs[0].shape) {
          const thisSize = this.shape.reduce((acc, e) => acc * e, 1);
          const srcsSize = srcs[0].shape.reduce((acc, e) => acc * e, 1);
          if (thisSize > srcsSize) {
            srcs[0].data = broadcast_to(srcs[0].data, this.shape);
          } else if (thisSize < srcsSize) {
            this.data = broadcast_to(this.data, srcs[0].shape);
          }
        }
        // @ts-ignore
        out = this.data.lt(srcs[0].data).selection.data.map(Number);
        break;
    }
    return new LazyBuffer(out);
  }

  // r(op: ReduceOps, new_shape: number[]): LazyBuffer {
  //   const DEBUG = 1; // Assuming a DEBUG constant; adjust its scope as needed
  //   if (DEBUG >= 1) console.log(op, this, new_shape);
  //   if (this.shape.length !== new_shape.length) {
  //     throw new Error("reduce shapes must have the same dimensions");
  //   }
  //   const axis: number[] = this.shape
  //     .map((s, i) => (s !== new_shape[i] ? i : -1))
  //     .filter((i) => i !== -1);
  //
  //   switch (op) {
  //     case "SUM":
  //       return new LazyBuffer(nj.sum(this.data, axis, false));
  //     case "MAX":
  //       // Since nj does not directly support reduce max with axis, use custom logic or consider extending nj or using a different library
  //       throw new Error(
  //         "ReduceOps.MAX is not directly supported, needs custom implementation",
  //       );
  //     default:
  //       throw new Error(`NotImplementedError: ${op}`);
  //   }
  // }

}

let a = new LazyBuffer(nj.array([0, 2, 4]));
let b = new LazyBuffer(nj.array([0, 3, 1]));

console.log(a.e("CMPLT", b));
