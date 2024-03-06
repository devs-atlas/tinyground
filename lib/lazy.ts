import { BinaryOps, LoadOps, UnaryOps } from "./ops";

import * as tf from "@tensorflow/tfjs";

class LazyBuffer {
  data: tf.Tensor;

  constructor(buf: tf.Tensor) {
    this.data = buf;
  }

  get realized(): tf.Tensor {
    return this.data;
  }

  get shape(): number[] {
    return this.data.shape;
  }

  static loadop(op: LoadOps, shape: number[], arg?: number): LazyBuffer {
    switch (op) {
      case "RAND":
        return new LazyBuffer(tf.randomUniform(shape));
      case "CONST":
        return new LazyBuffer(tf.ones(shape).mul(arg || 1));
      case "EMPTY":
        return new LazyBuffer(tf.zeros(shape));
      default:
        throw new Error("Can only load RAND, CONST, or EMPTY Loadop.");
    }
  }

  e(op: UnaryOps | BinaryOps, ...srcs: LazyBuffer[]): LazyBuffer {
    let out = this.data;
    switch (op) {
      case "NEG":
        out = tf.neg(out);
        break;
      case "EXP2":
        out = tf.pow(out, 2);
        break;
      case "LOG2":
        out = tf.log(out).div(tf.log(2));
        break;
      case "SIN":
        out = tf.sin(out);
        break;
      case "SQRT":
        out = tf.sqrt(out);
        break;
      case "ADD":
        out = this.data.add(srcs[0].data);
        break;
      case "SUB":
        out = this.data.sub(srcs[0].data);
        break;
      case "MUL":
        out = this.data.mul(srcs[0].data);
        break;
      case "DIV":
        out = this.data.div(srcs[0].data);
        break;
      case "MAX":
        out = this.data.maximum(srcs[0].data);
        break;
      case "CMPLT":
        out = this.data.less(srcs[0].data);
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

let a = new LazyBuffer(tf.tensor([0, 2, 4]));
let b = new LazyBuffer(tf.tensor([0, 3, 1]));

a.e("CMPLT", b).data.print();
