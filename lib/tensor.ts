import * as tf from "@tensorflow/tfjs";
import LazyBuffer from "./lazy";
import * as mlops from "./mlops";

type NDArray = number[] | NDArray[];

import Fn from "./fn";
import { ReduceOps } from "./ops";

export class Tensor {
  grad?: Tensor;
  data: LazyBuffer;
  shape: number[];
  requires_grad: boolean;
  context?: Fn;

  constructor(data: number | NDArray | tf.Tensor | LazyBuffer, requires_grad: boolean = false) {
    if (data instanceof tf.Tensor) {
      this.data = new LazyBuffer(data);
      this.shape = data.shape;
    } else if (typeof data == "number") {
      this.data = new LazyBuffer(tf.tensor([data]));
      this.shape = [];
    } else if (Array.isArray(data)) {
      this.data = new LazyBuffer(tf.tensor(data));
      this.shape = this.data.shape;
    } else {
      this.data = data;
      this.shape = data.shape;
    }
    this.requires_grad = requires_grad;
  }

  get dtype() {
    return this.data.dtype;
  }

  static full(shape: number[], fill_value: number, requires_grad: boolean) {
    return new Tensor(tf.ones(shape).mul(fill_value), requires_grad);
  }

  static ones(shape: number[], requires_grad: boolean) {
    return new Tensor(tf.ones(shape), requires_grad);
  }

  static zeros(shape: number[], fill_value: number, requires_grad: boolean) {
    return new Tensor(tf.zeros(shape).mul(fill_value), requires_grad);
  }

  full_like(fill_value: number, dtype?: string) {
    return Tensor.full(this.shape, fill_value, true)
  }

  add(tensor: Tensor) {
    return mlops.Add.run_op([this, tensor]);
  }

  sub(tensor: Tensor) {
    return mlops.Sub.run_op([this, tensor]);
  }

  mul(tensor: Tensor) {
    return mlops.Mul.run_op([this, tensor]);
  }

  _reduce(op: ReduceOps, axis?: number[] | number, keepdim = false): Tensor {
    let axis_: number[];
    if (axis === undefined) {
      axis_ = Array.from({ length: this.shape.length }, (_, index) => index);
    } else if (typeof axis === "number") {
      axis_ = [axis];
    } else {
      axis_ = axis;
    }

    // @ts-ignore
    let reducedShape = this.shape.filter((_, index) => !axis_.includes(index));

    if (reducedShape.includes(0) && !this.shape.includes(0)) {
      if (keepdim) {
        reducedShape = reducedShape.map((axis) => (axis ? axis : 1));
      }
      // TODO: fix that
      const fillVal = op === 'SUM'  ? 0 : Infinity;
      return Tensor.full(reducedShape, fillVal, this.requires_grad);
    }

    const new_shape = this.shape.map((s, i) => axis_.includes(i) ? 1 : s)

    let ret =
      op === 'SUM'
        ? mlops.Sum.run_op([this], {new_shape})
        : mlops.Max.run_op([this], {new_shape});

    return keepdim ? ret : ret.reshape(reducedShape);
  }

  //TODO: do i need to restate the axis type here and _reduce? should only be needed on outward facing methods.
  sum(axis?: number | number[], keepdim = false) {
    // @ts-ignore
    return this._reduce(mlops.Sum, axis, keepdim);
  }
  max(axis?: number | number[], keepdim = false) {
    // @ts-ignore
    return this._reduce(mlops.Max, axis, keepdim);
  }
  min(axis?: number | number[], keepdim = false) {
    return -this.neg().max((axis = axis), (keepdim = keepdim));
  }

  // mlops (unary)

  neg() {
    return mlops.Neg.run_op([this]);
  }
  contiguous() {
    return mlops.Contiguous.run_op([this]);
  }
  contiguous_backward() {
    return mlops.ContiguousBackward.run_op([this]);
  }
  log() {
    return mlops.Log.run_op([this]);
  }
  // log2() {
  //   return mlops.Log.run_op([this]).div();
  // }

  // broadcasted binary mlops

  // _broadcasted(y: Tensor | number, reverse: boolean = false) {
  //   let x: Tensor = this;
  //   if (!(y instanceof Tensor)) {
  //     if (this.shape.includes(0)) {
  //       return this, this.full_like(y);
  //     }
  //     //TODO: dtype here
  //     y = new Tensor(y, false);
  //   }
  //
  //   [x, y] = reverse ? [y, x] : [x, y];
  //
  //   let xshape = x.shape;
  //   let yshape = y.shape;
  //
  //   if (xshape === yshape) {
  //     return [x, y];
  //   }
  //
  //   let shape_delta = xshape.length - yshape.length;
  //   if (shape_delta > 0) {
  //     const newShape = new Array(shape_delta).fill(1); // Create an array of `shape_delta` ones
  //     y = y.reshape([...newShape, ...y.shape]); // Spread the new dimensions and original shape
  //   } else if (shape_delta < 0) {
  //     const newShape = new Array(shape_delta).fill(-1); // Create an array of `shape_delta` ones
  //     x = x.reshape([...newShape, ...y.shape]); // Spread the new dimensions and original shape
  //   }
  //   xshape = x.shape;
  //   //@ts-ignore
  //   yshape = y.shape;
  //   if (xshape == yshape) {
  //     return [x, y];
  //   }
  //
  //   //do this
  //   let shape_ret = xshape.map((x, i) => Math.max(x, yshape[i]));
  //
  //   if (xshape !== shape_ret) {
  //     x = x.expand(shape_ret);
  //   }
  //   if (yshape !== shape_ret) {
  //     y = y.expand(shape_ret);
  //   }
  //   return [x, y];
  // }

  // movement mlops

  reshape(shape: number | (number | null)[]) {
    if (typeof shape === 'number')
      shape = [shape];
    if (shape.filter((e) => e === -1).length > 1)
      throw new Error("At most one dimension of shape can be -1")

    for (let i = 0; i < shape.length; ++i) {
      if (shape[i] === -1) {
        shape[i] = this.shape.reduce((a, b) => a * b, 1);
      } else if (shape[i] === null) {
        shape[i] = this.shape[i]
      }
    }
    return mlops.Reshape.run_op([this], { shape });
  }

  toString() {
    let repr = `Data: ${this.data.toString()}`;
    if (this.requires_grad) {
      repr += `, grad: ${this.grad ? this.grad.data : undefined}`;
    }
    return repr;
  }
}
