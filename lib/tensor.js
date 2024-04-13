import * as tf from "@tensorflow/tfjs";
import LazyBuffer from "./lazy";
import * as mlops from "./mlops";

import Fn from "./fn";

function isNDArray(data) {
  if (Array.isArray(data)) {
    return data.every(
      (element) => typeof element === "number" || isNDArray(element)
    );
  }
  return false;
}

export default class Tensor {
  grad;
  data;
  shape;
  requires_grad;
  context;

  // Tensor offers four initialization options: number, multi-dimensional array, tf.Tensor, or a LazyBuffer
  constructor(data, requires_grad = false) {
    if (data instanceof tf.Tensor) {
      this.data = new LazyBuffer(data);
    } else if (typeof data == "number") {
      this.data = new LazyBuffer(tf.tensor([data]));
    } else if (isNDArray(data)) {
      this.data = new LazyBuffer(tf.tensor(data));
    } else {
      this.data = data;
    }
    this.shape = this.data.shape;
    this.requires_grad = requires_grad;
  }

  get dtype() {
    return this.data.dtype;
  }

  static full(shape, fill_value, requires_grad) {
    return new Tensor(tf.ones(shape).mul(fill_value), requires_grad);
  }

  static ones(shape, requires_grad) {
    return new Tensor(tf.ones(shape), requires_grad);
  }

  static zeros(shape, fill_value, requires_grad) {
    return new Tensor(tf.zeros(shape).mul(fill_value), requires_grad);
  }

  full_like(fill_value, dtype) {
    return Tensor.full(this.shape, fill_value, true);
  }

  static argfix(x) {
    return x instanceof Tensor ? x : new Tensor(x);
  }

  add(addend) {
    return mlops.Add.run_op([this, Tensor.argfix(addend)]);
  }

  sub(minuend) {
    return mlops.Sub.run_op([this, Tensor.argfix(minuend)]);
  }

  mul(factor) {
    return mlops.Mul.run_op([this, Tensor.argfix(factor)]);
  }

  div(divisor) {
    return mlops.Div.run_op([this, Tensor.argfix(divisor)]);
  }

  // dot(w: Tensor) {
  //   let n1 = this.shape.length;
  //   let n2 = w.shape.length;
  //
  //
  // }

  /*
   * We need _reduce to systematize the argument handling for all reduce ops,
   * giving us a convenient way to perform these reduction operations
   * under a common interface(_reduce).
   */
  _reduce(op, axis, keepdim = false) {
    let axis_;
    //if no axis, reduce along all.
    if (axis === undefined) {
      axis_ = Array.from({ length: this.shape.length }, (_, index) => index);
    } else if (typeof axis === "number") {
      axis_ = [axis];
    } else {
      axis_ = axis;
    }

    /*
     * The new shape will be the original shape but the axes
     * specified in axis argument are omitted because they're reduced.
     */
    let reducedShape = this.shape.filter((_, index) => !axis_.includes(index));

    //TODO: Figure out this line - why would reducedShape have 0?
    if (reducedShape.includes(0) && !this.shape.includes(0)) {
      if (keepdim) {
        reducedShape = reducedShape.map((axis) => (axis ? axis : 1));
      }
      const fillVal = op === "SUM" ? 0 : Infinity;
      return Tensor.full(reducedShape, fillVal, this.requires_grad);
    }

    //the new_shape will have 1 in place of the original shape in
    //axes that are reduced
    const new_shape = this.shape.map((s, i) => (axis_.includes(i) ? 1 : s));

    let ret =
      op === "SUM"
        ? mlops.Sum.run_op([this], { new_shape })
        : mlops.Max.run_op([this], { new_shape });

    //if keepdim, reduced axes will be 1, otherwise just gone
    return keepdim ? ret : ret.reshape(reducedShape);
  }

  sum(axis, keepdim = false) {
    return this._reduce("SUM", axis, keepdim);
  }
  max(axis, keepdim = false) {
    return this._reduce(mlops.Max, axis, keepdim);
  }
  min(axis, keepdim = false) {
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

  permute(order) {
    return mlops.Permute.run_op([this], order);
  }

  //-1 indexing --> one dim can be -1, this dim will be leftover elements
  reshape(shape) {
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

  //-1 indexing --> for these indices, dont expand and use original shape
  expand(shape) {
    return mlops.Expand.run_op(
      [this],
      shape.map((x, i) => (x === -1 ? this.shape[i] : x))
    );
  }

  transpose(axis1 = 1, axis2 = 0) {
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

  deepwalk() {
    const _dfs = (node, seen, nodes) => {
      seen.add(node);
      for (let i of node.context?.parents ?? []) {
        if (!seen.has(i)) _dfs(i, seen, nodes);
        nodes.push(node);
      }
      return nodes;
    };
    return _dfs(this, new Set(), []);
  }

  /*
   * can only backprop on a scalar...
   *
   * an mlop backward function can either return a single gradient(needs to be wrapped)or a list of gradients.
   *
   * the gradients from mlops are LazyBuffer, so turn into Tensor (grad doesn't require grad)
   * if the gradient returned is undefined, set that entry to undefined.
   *
   * to call backward in an mlop, the grad_output being fed in from node ahead
   * must have defined grad
   *
   * these list of gradients correspond to the parents in _ctx.
   *
   * there are two cases where we don't want to pass the gradients back:
   * 1. mlops backward returns undefined
   * 2. the parent doesn't require_grad
   *
   * otherwise, pass back gradients --> if the gradient is defined for the parent, accumulate.
   */
  backward() {
    console.log(`have shape ${this.shape}`);
    if (this.shape.length !== 1 || this.shape[0] !== 1) {
      throw new Error("can only backprop on scalars");
    }

    this.grad = new Tensor(1.0, false);

    for (let node of this.deepwalk().reverse()) {
      //TODO: Is this right? I think we should still walk through
      if (node.grad === undefined)
        throw new Error("cannot deepwalk node with undefined grad");

      if (node.context?.parents === undefined)
        throw new Error("cannot deepwalk node with undefined parents");

      let grads = node.context?.backward(node.grad.data);
      let _grads = (grads instanceof LazyBuffer ? [grads] : grads)
        .map((g) => {
          if (g !== undefined) return new Tensor(g, false);
        })
        .filter(Boolean);

      for (let i = 0; i < _grads.length; ++i) {
        let [t, g] = [node.context?.parents[i], _grads[i]];

        if (!t?.requires_grad) continue;

        // shapes must be equal
        if (!g.shape.every((e, i) => e === t.shape[i])) {
          throw new Error(
            `Grade shape ${g.shape} must match tensor shape ${t.shape}`
          );
        }

        t.grad = t.grad === undefined ? g : t.grad.add(g);
      }
      // delete context
      node.context = undefined;
    }
    return this;
  }
}
