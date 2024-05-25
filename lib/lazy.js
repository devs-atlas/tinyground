import * as tf from "@tensorflow/tfjs";

export default class LazyBuffer {
  data;

  constructor(buf) {
    this.data = buf;
  }

  get realized() {
    return this.data;
  }

  get shape() {
    return this.data.shape;
  }

  get dtype() {
    return this.data.dtype;
  }

  //TODO: why does teeny continuous have x argument here?
  contiguous() {
    return this;
  }

  const(fill_value) {
    return new LazyBuffer(tf.ones(this.shape).mul(fill_value));
  }

  static loadop(op, shape, arg) {
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

  e(op, ...srcs) {
    //add debug printing
    let out = this.data;

    switch (op) {
      case "NEG":
        out = tf.neg(out);
        break;
      case "EXP":
        console.log(Math.E);
        out = tf.pow(Math.E, out);
        break;
      case "EXP2":
        out = tf.pow(2, out);
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

    //TODO: check out TeenyGrad code for this - they use highest order type in self or srcs
    return new LazyBuffer(out);
  }

  r(op, new_shape) {
    if (this.shape.length !== new_shape.length) {
      throw new Error("Reduce shapes must have same dimensions");
    }

    const axes = [];
    for (let i = 0; i < this.shape.length; i++) {
      if (this.shape[i] != new_shape[i]) {
        axes.push(i);
      }
    }

    if (op === "SUM") {
      return new LazyBuffer(tf.sum(this.data, axes, true));
    } else if (op === "MAX") {
      return new LazyBuffer(tf.max(this.data, axes, true));
    } else {
      throw new Error("Reduce operation must be SUM or MAX");
    }
  }

  cast(dtype) {
    //it doesn't let me use string for dtype
    //@ts-ignore
    return new LazyBuffer(this.data.cast(dtype));
  }

  // TODO: Allow for the fancy indexing of teenygrad - None, not specifying certain dims, just a number...
  // TODO: Can I give a better name than arg
  reshape(arg) {
    return new LazyBuffer(this.data.reshape(arg));
  }

  expand(arg) {
    return new LazyBuffer(tf.broadcastTo(this.data, arg));
  }

  shrink(arg) {
    const starts = arg.map((p) => p[0]);
    const sizes = arg.map((p) => p[1] - p[0]);
    const resultTensor = tf.slice(this.data, starts, sizes);
    return new LazyBuffer(resultTensor);
  }

  permute(arg) {
    return new LazyBuffer(this.data.transpose(arg));
  }

  pad(arg) {
    return new LazyBuffer(tf.pad(this.data, arg));
  }

  stride(arg) {
    const begin = new Array(this.shape.length).fill(0);
    const end = this.shape;
    const strides = arg;
    return new LazyBuffer(this.data.stridedSlice(begin, end, strides));
  }

  toString() {
    return `LazyBuffer: ${this.shape}, ${this.dtype}`;
  }
}
