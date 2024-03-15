import * as tf from "@tensorflow/tfjs";
import * as mlops from "./mlops";
import LazyBuffer from "./lazy";
import { BinaryOps, LoadOps, UnaryOps, ReduceOps } from "./ops";
export class Tensor {
  grad?: Tensor;
  data: LazyBuffer;
  shape: number[];
  requires_grad: boolean;
  context?: Fn;

  constructor(data: number | tf.Tensor | LazyBuffer, requires_grad: boolean) {
    if (data instanceof tf.Tensor) {
      this.data = new LazyBuffer(data);
      this.shape = data.shape;
    } else if (typeof data == "number") {
      this.data = new LazyBuffer(tf.tensor([data]));
      this.shape = [];
    } else if (Array.isArray(data)) {
      this.data = new LazyBuffer(tf.tensor(data));
      this.shape = data.shape;
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

  _reduce(fxn: Fn, axis?: number[] | number, keepdim = false): Tensor {
    if (!(fxn instanceof mlops.Sum || fxn instanceof mlops.Max)) {
      throw new Error("fxn must be an instance of Sum or Max");
    }
    let axis_ = axis;
    if (!axis_) {
      axis_ = Array.from({ length: this.shape.length }, (_, index) => index);
    } else if (typeof axis_ === "number") {
      axis_ = [axis_];
    }

    //@ts-ignore
    let reducedShape = this.shape.filter((_, index) => !axis_.includes(index));

    if (reducedShape.includes(0) && !this.shape.includes(0)) {
      if (keepdim) {
        reducedShape = reducedShape.map((axis) => (axis ? axis : 1));
      }
      const fillVal = fxn instanceof mlops.Sum ? 0 : Infinity;
      return Tensor.full(reducedShape, fillVal, this.requires_grad);
    }

    let ret =
      fxn instanceof mlops.Sum
        ? mlops.Sum.run_op([this], { axis: axis_, keepdim })
        : mlops.Max.run_op([this], { axis: axis_, keepdim });

    return keepdim ? ret : ret.reshape(reducedShape);
  }

  //TODO: do i need to restate the axis type here and _reduce? should only be needed on outward facing methods.
  sum(axis?: number | number[], keepdim = false) {
    //TODO: is this a temporary fix - why can't I pass mlops.Sum in directly
    return this._reduce(mlops.Sum as unknown as Fn, axis, keepdim);
  }
  max(axis?: number | number[], keepdim = false) {
    return this._reduce(mlops.Max as unknown as Fn, axis, keepdim);
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

  _broadcasted(y: Tensor | number, reverse: boolean = false) {
    let x: Tensor = this;
    if (!(y instanceof Tensor)) {
      if (this.shape.includes(0)) {
        return this, this.full_like(y);
      }
      //TODO: dtype here
      y = new Tensor(y, false);
    }

    [x, y] = reverse ? [y, x] : [x, y];

    let xshape = x.shape;
    let yshape = y.shape;

    if (xshape === yshape) {
      return [x, y];
    }

    let shape_delta = xshape.length - yshape.length;
    if (shape_delta > 0) {
      const newShape = new Array(shape_delta).fill(1); // Create an array of `shape_delta` ones
      y = y.reshape([...newShape, ...y.shape]); // Spread the new dimensions and original shape
    } else if (shape_delta < 0) {
      const newShape = new Array(shape_delta).fill(-1); // Create an array of `shape_delta` ones
      x = x.reshape([...newShape, ...y.shape]); // Spread the new dimensions and original shape
    }
    xshape = x.shape;
    //@ts-ignore
    yshape = y.shape;
    if (xshape == yshape) {
      return [x, y];
    }

    //do this
    let shape_ret = xshape.map((x, i) => Math.max(x, yshape[i]));

    if (xshape !== shape_ret) {
      x = x.expand(shape_ret);
    }
    if (yshape !== shape_ret) {
      y = y.expand(shape_ret);
    }
    return [x, y];
  }

  // movement mlops

  reshape(shape: number | number[]) {
    shape =
    return mlops.Reshape.run_op([this], { shape });
  }

  toString() {
    let repr = `Data: ${this.data}`;
    if (this.requires_grad) {
      repr += `, grad: ${this.grad ? this.grad.data : undefined}`;
    }
    return repr;
  }
}

export class Fn {
  needs_input_grad: boolean[];
  requires_grad: boolean;
  parents?: Tensor[];

  constructor(tensors: Tensor[]) {
    this.needs_input_grad = tensors.map((t) => t.requires_grad);
    this.requires_grad = this.needs_input_grad.some(Boolean);
    if (this.requires_grad) {
      this.parents = tensors;
    }
  }

  forward(_: any, ...__: any): any { }
  backward(_: any, ...__: any): any { }

  static run_op(tensors: Tensor[], options = {}): Tensor {
    // TODO: can we just make this "Fn"?
    const context = new this(tensors);
    const tensor = new Tensor(
      context.forward(
        tensors.map((t) => t.data),
        options,
      ),
      context.requires_grad,
    );
    if (context.requires_grad) {
      tensor.context = context;
    }
    return tensor;
  }
}
