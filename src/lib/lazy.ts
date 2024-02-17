import { NdArray as NdA, default as nj } from "@d4c/numjs";
import { UnaryOps, BinaryOps, ReduceOps, TernaryOps, LoadOps } from './ops';

//Op type should be union of types in ops

class LazyBuffer {
  _nj: NdA;
  constructor(buf: NdA) {
    this._nj = buf;
  }
  get shape(): number[] {
    return this._nj.shape;
  }

  // static function loadop(op: Op, shape: number[], arg?: number): LazyBuffer | undefined {
  //   if (op == LoadOps.RAND) {
  //     return new LazyBuffer(nj.random(shape))
  //   }
  //   if (op == LoadOps.CONST) {
  //     return new LazyBuffer(nj.ones(shape).multiply(arg))
  //   }
  //   // should i do this??? 
  //   // if (op == LoadOps.EMPTY) {
  //   //   return new LazyBuffer(nj.random(shape))
  //   // }
  //   else {
  //     //improve
  //     console.log("ERROR")
  //   }
  // }

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
