import { NdArray as NdA, default as nj } from "@d4c/numjs";
import ops from "ndarray-ops";
import { LoadOps } from "./ops";

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
      case LoadOps.RAND:
        return new LazyBuffer(nj.random(shape));
      case LoadOps.CONST:
        return new LazyBuffer(nj.ones(shape).multiply(arg || 1));
      case LoadOps.EMPTY:
        return new LazyBuffer(nj.empty(shape));
      default:
        throw new Error("Can only load RAND, CONST, or EMPTY Loadop.");
    }
  }

  e(op: UnaryOps | BinaryOps | TernaryOps, ...srcs: LazyBuffer[]): LazyBuffer {
    let out = this.data;
    switch (op) {
      case UnaryOps.NEG:
        out = nj.negative(out);
        break;
      case UnaryOps.EXP2:
        out = nj.power(out, 2);
        break;
      case UnaryOps.LOG2:
        out = nj.log(out).divide(nj.log(2));
        break;
      case UnaryOps.SIN:
        out = nj.sin(out);
        break;
      case UnaryOps.SQRT:
        out = nj.sqrt(out);
        break;
      case BinaryOps.ADD:
        out = nj.add(this.data, srcs[0].data);
        break;
      case BinaryOps.SUB:
        out = nj.subtract(this.data, srcs[0].data);
        break;
      case BinaryOps.MUL:
        out = nj.multiply(this.data, srcs[0].data);
        break;
      case BinaryOps.DIV:
        out = nj.divide(this.data, srcs[0].data);
        break;
      case BinaryOps.MAX:
        let a = ndarray(this.data.selection.data);
        let b = ndarray(srcs[0].data.selection.data);

        // TODO: probably a shallow copy, not what we want
        let dest = a;
        ops.max(dest, a, b);

        return new LazyBuffer(new nj.NdArray(dest, this.data.shape));
    }
  }

  // function e(op: Op, srcs: LazyBuffer[]){
  //   if (op==UnaryOps.NEG){
  //     return this._nj.multiply(-1)
  //   }
  //   if (op==UnaryOps.EXP2){
  //     return this._nj.pow(2)
  //   }
  //   if (op==UnaryOps.NEG){
  //     return this._nj.log(-1)
  //   }
  // }
}

let dest = ndarray([1, 2, 9, 4]);

let b = ndarray([1, 2, 3, 420]);
let c = ndarray([1, 2, 6, 69]);
ops.max(dest, b, c);
console.log(dest);
