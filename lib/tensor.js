import * as tf from "@tensorflow/tfjs";
import LazyBuffer from "./lazy";
import * as mlops from "./mlops";

//TODO: look into tidy and dispose - necessary somewhere fs

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

  get scalar() {
    return this.shape.reduce((acc, x) => (acc *= x)) === 1;
  }

  get vector() {
    return this.shape.filter(dim => dim > 1).length === 1;
  }

  static full(shape, fill_value, requires_grad) {
    return new Tensor(tf.ones(shape).mul(fill_value), requires_grad);
  }

  static ones(shape, requires_grad) {
    return new Tensor(tf.ones(shape), requires_grad);
  }

  static zeros(shape, requires_grad) {
    return new Tensor(tf.zeros(shape), requires_grad);
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

  /*
   * how to dot:
   * 1. dimensions must match
   * if w is a vector, the last dimension of self must match the last dimension of w
   * if w is a matrix, the last dimension of self must match the second to last dimension
   * 2. add 1 dim to x and w if necessary(either is vector)
   * 2a. for x, 1 dim is second to last
   * 2b. for w, 1 dim is third
   * 3. if w is a matrix, switch last two dimensions to element-wise multiply rows and columns
   * 4. perform element-wise multiplication and sum
   */
  dot(w) {
    const this_dims = this.shape.length;
    const w_dims = w.shape.length;

    let x = this;

    if (x.scalar || w.scalar)
      throw new Error("Cannot dot scalar");
    if (w.vector && w.shape[w_dims - 1] != x.shape[this_dims - 1])
      throw new Error("When w is a vector, the last dimensions must match");
    if (!w.vector && w.shape[w_dims - 2] != x.shape[this_dims - 1])
      throw new Error("Inner dimensions must match!")
    if (!x.vector && !w.vector) {
      //insert one dim in second to last dimension of x
      const newXShape = [...x.shape.slice(0, -1), 1, x.shape[this_dims - 1]];
      x = x.reshape(newXShape);
      //insert one dim in third to last dim
      const newWShape = [...w.shape.slice(0, -2), 1, ...w.shape.slice(-2)];
      console.log(newWShape);
      w = w.reshape(newWShape).transpose(1, 2);
      console.log(w.shape);
    }
    return (x.mul(w)).sum(-1);
  }

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

    //Negative indexing
    axis_ = axis_.map((el) => (el >= 0 ? el : this.shape.length + el))

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

    // new_shape will have 1 in place of the original shape in axes that are reduced
    // this is convention - summing along all axes with `keepdim = false` returns
    // a shape of [1]
    let new_shape;
    if (axis === undefined && !keepdim) new_shape = [1];
    else new_shape = this.shape.map((s, i) => (axis_.includes(i) ? 1 : s));

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
        let prod = shape.reduce((acc, dim) => dim === -1 ? acc : (acc * dim), 1);
        shape[i] = this.shape.reduce((acc, dim) => acc * dim) / prod;
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
      shape.map((x, i) => (x == -1 ? this.shape[i] : x))
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
    let repr = `Data: ${this.data.toString()} `;
    if (this.requires_grad) {
      repr += `, grad: ${this.grad ? this.grad.data : undefined} `;
    }
    return repr;
  }

  deepwalk() {
    const _deepwalk = (node, visited, nodes) => {
      visited.add(node);
      if (node.context) {
        for (let parent of node.context.parents) {
          if (!visited.has(parent)) {
            _deepwalk(parent, visited, nodes);
          }
        }
        nodes.push(node);
      }
      return nodes;
    };

    return _deepwalk(this, new Set(), []);
  }

  /*
  how to backprop:
  0. you can only backprop on a scalar
  1. derivative w.r.t itself is 1 - implicit gradient creation
  2. walk through computation graph starting at first node (which should have grad 1)
  2a. make sure current gradient isn't none - you can't pass back the gradient if
      the gradient is undefined
  3. call backward on current nodes ctx
     backward on mlops returns a list of gradients which can either be LB or undefined
     undefined means that the corresponding tensor used in the op doesn't require grad
     this is determined in the constructor of Fn with the needs_input_grad property
  4. cast each gradient returned from ctx backward to either a Tensor or undefined
     depending on whether or not the gradient was an LB or undefined
  5. loop through gradients and parents - each (gradient,parent) pair corresponds to the
     gradient for that particular parent in the operation
  6. if the gradient isn't undefined, the parent requires grad(i think these always match?),
     and the shape of the gradient and parent match, pass down the gradient to the parent 
     if the parent already has a gradient, accumulate
     otherwise(parent's grad is undefined), just assign the gradient
  7. delete context so next gradient-setting operation can overwrite it
  8. return 'this' once backprop is done
   
  TODO: am i sure that i should have to check if context is undefined?
  it makes sense -- once you reach a node 
  
  */
  backward() {
    if (!this.scalar) {
      throw new Error("can only backprop on scalars");
    }
    //1
    this.grad = new Tensor(1.0, false);
    //2
    for (let node of this.deepwalk().reverse()) {
      //2a
      if (node.grad === undefined)
        throw new Error("cannot deepwalk node with undefined grad");
      //3
      let grads = node.context.backward(node.grad.data);
      //4
      grads = grads.map((g) => { return g != undefined ? new Tensor(g, false) : undefined });
      //5
      let parents = node.context.parents;
      for (let i = 0; i < parents.length; ++i) {
        //6
        if (grads[i] != undefined && parents[i].requires_grad) { //double checking 
          if (!grads[i].shape.every((e, ix) => e === parents[i].shape[ix])) {
            throw new Error(`grads shape ${g.shape} must match tensor shape ${t.shape} `);
          }
          parents[i].grad = parents[i].grad === undefined ? grads[i] : parents[i].grad.add(grads[i]);
        }
      }
      //7
      node.context = undefined;
    }
    //8
    return this;
  }
}
