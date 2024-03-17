import * as tf from "@tensorflow/tfjs";
import LazyBuffer from "./lazy";
import * as mlops from "./mlops";

import Fn from "./fn";
import { ReduceOps } from "./ops";
import { type NDArray, isNDArray } from "./utils";

export default class Tensor {
  grad?: Tensor;
  data: LazyBuffer;
  shape: number[];
  requires_grad: boolean;
  context?: Fn;

  constructor(
    data: number | NDArray | tf.Tensor | LazyBuffer,
    requires_grad: boolean = false
  ) {
    if (data instanceof tf.Tensor) {
      this.data = new LazyBuffer(data);
      this.shape = data.shape;
    } else if (typeof data == "number") {
      this.data = new LazyBuffer(tf.tensor([data]));
      this.shape = [];
    } else if (isNDArray(data)) {
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
    return Tensor.full(this.shape, fill_value, true);
  }

  static argfix(x: number | Tensor) {
    return x instanceof Tensor ? x : new Tensor(x);
  }

  add(addend: Tensor | number) {
    return mlops.Add.run_op([this, Tensor.argfix(addend)]);
  }

  sub(minuend: Tensor | number) {
    return mlops.Sub.run_op([this, Tensor.argfix(minuend)]);
  }

  mul(factor: Tensor | number) {
    return mlops.Mul.run_op([this, Tensor.argfix(factor)]);
  }

  div(divisor: Tensor | number) {
    return mlops.Div.run_op([this, Tensor.argfix(divisor)]);
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
      const fillVal = op === "SUM" ? 0 : Infinity;
      return Tensor.full(reducedShape, fillVal, this.requires_grad);
    }

    const new_shape = this.shape.map((s, i) => (axis_.includes(i) ? 1 : s));

    let ret =
      op === "SUM"
        ? mlops.Sum.run_op([this], { new_shape })
        : mlops.Max.run_op([this], { new_shape });

    return keepdim ? ret : ret.reshape(reducedShape);
  }

  //TODO: do i need to restate the axis type here and _reduce? should only be needed on outward facing methods.
  sum(axis?: number | number[], keepdim = false) {
    // @ts-ignore
    return this._reduce("SUM", axis, keepdim);
  }
  max(axis?: number | number[], keepdim = false) {
    // @ts-ignore
    return this._reduce(mlops.Max, axis, keepdim);
  }
  min(axis?: number | number[], keepdim = false) {
    return this.neg().max(axis, keepdim).neg();
  }

  // Unary mlops
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
  log2() {
    return mlops.Log.run_op([this]).div(Math.log(2));
  }
  exp() {
    return mlops.Exp.run_op([this]);
  }
  exp2() {
    return mlops.Exp.run_op([this.mul(Math.log(2))]);
  }
  relu() {
    return mlops.Relu.run_op([this]);
  }
  sigmoid() {
    return mlops.Sigmoid.run_op([this]);
  }
  sqrt() {
    return mlops.Sqrt.run_op([this]);
  }
  rsqrt() {
    return new Tensor([1]).div(mlops.Sqrt.run_op([this]));
  }
  sin() {
    return mlops.Sin.run_op([this]);
  }
  cos() {
    return this.neg()
      .add(Math.PI / 2)
      .sin();
  }
  tan() {
    return this.sin().div(this.cos());
  }

  permute(order: number[]) {
    return mlops.Permute.run_op([this], order);
  }

  reshape(shape: number | (number | null)[]) {
    if (typeof shape === "number") shape = [shape];
    if (shape.filter((e) => e === -1).length > 1)
      throw new Error("At most one dimension of shape can be -1");

    for (let i = 0; i < shape.length; ++i) {
      if (shape[i] === -1) {
        shape[i] = this.shape.reduce((a, b) => a * b, 1);
      } else if (shape[i] === null) {
        shape[i] = this.shape[i];
      }
    }
    return mlops.Reshape.run_op([this], { shape });
  }

  expand(shape: number[]) {
    return mlops.Expand.run_op(
      [this],
      shape.map((x, i) => (x === -1 ? this.shape[i] : x))
    );
  }

  transpose(axis1 = 1, axis2 = 0): Tensor {
    let order = [...Array(this.shape.length)].map((_, i) => i);
    [order[axis1], order[axis2]] = [order[axis2], order[axis1]];
    return this.permute(order);
  }

  get T() {
    return this.transpose();
  }

  toString() {
    let repr = `Data: ${this.data.toString()}`;
    if (this.requires_grad) {
      repr += `, grad: ${this.grad ? this.grad.data : undefined}`;
    }
    return repr;
  }
}
