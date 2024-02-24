import { NdArray as NdA, default as nj } from "@d4c/numjs";
import ops from "ndarray-ops";
import { BinaryOps, LoadOps, UnaryOps } from "./ops";

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
NdA.prototype.lt = function (x) {
  const arr = this.clone();
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
        // @ts-ignore
        out = this.data.emax(srcs[0].data);
        break;
      case "CMPLT":
        // @ts-ignore
        out = this.data.lt(srcs[0].data).selection.data.map(Number);
        break;
    }
    return new LazyBuffer(out);
  }
}

let a = new LazyBuffer(nj.array([0, 2, 4]));
let b = new LazyBuffer(nj.array([0, 3, 1]));

console.log(a.e("CMPLT", b));
