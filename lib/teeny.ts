import * as tf from "@tensorflow/tfjs";

export class Tensor {
  grad?: Tensor;
  data: tf.Tensor;
  shape: number[];
  requires_grad: boolean;
  context?: Fn;

  // TODO: add kwargs for this - check og teeny
  full(shape: number[], fill_value: number, requires_grad: boolean) {
    // TODO: add expand
    return new Tensor(fill_value, requires_grad = requires_grad).expand(shape)
  }

  constructor(data: number | tf.Tensor, requires_grad: boolean) {
    if (data instanceof tf.Tensor) {
      this.data = data;
      this.shape = data.shape;
    } else {
      this.data = tf.tensor([data]);
      this.shape = [];
    }
    this.requires_grad = requires_grad;
  }

  add(tensor: Tensor) {
    return Add.run_op([this, tensor]);
  }

  sub(tensor: Tensor) {
    return Sub.run_op([this, tensor]);
  }

  mul(tensor: Tensor) {
    return Mul.run_op([this, tensor]);
  }


  _reduce(fxn: Fn, axis?: number[] | number, keepdim = false) {
    let axis_ = axis;
    if (!axis_) {
      axis_ = Array.from({ length: this.shape.length }, (_, index) => index);
    }
    else if (typeof (axis_) === 'number') {
      axis_ = [axis_];
    }

    const reducedShape = this.shape.filter((_, index) => !axis_.includes(index));

    if (reducedShape.includes(0) && !this.shape.includes(0)) {
      return Tensor.full(this.shape.map((s, _) => s == 0 ? 1 : 0)
    }

    const ret = fxn.run_op(this, { new_shape?: number[] | number })
  }

  toString() {
    let repr = `Data: ${this.data}`;
    if (this.requires_grad) {
      repr += `, grad: ${this.grad ? this.grad.data : undefined}`;
    }
    return repr;
  }
}

class Fn {
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

class Expand extends Fn {
  input_shape!: number[];

  forward([x]: tf.Tensor[], { new_shape }: { new_shape: number[] }) {
    this.input_shape = x.shape;
    return x.broadcastTo(new_shape);
  }
  backward(grad_output: tf.Tensor) {
    // fix this - need to pass axis to
    // return sum(grad_output, this.input_shape);
  }
}

class Add extends Fn {
  forward([x, y]: tf.Tensor[]) {
    return x.add(y);
  }
  backward(grad_output: tf.Tensor) {
    // o = x + y
    // dx/do = 1 -> multiply by grad_output to represent dx/dL
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output : undefined,
    ];
  }
}

class Sub extends Fn {
  forward([x, y]: tf.Tensor[]) {
    return x.sub(y);
  }
  backward(grad_output: tf.Tensor) {
    // o = x - y = x - 1y
    // dy/do = -1
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output.neg() : undefined,
    ];
  }
}

class Mul extends Fn {
  x!: tf.Tensor;
  y!: tf.Tensor;

  forward([x, y]: tf.Tensor[]) {
    this.x = x;
    this.y = y;
    return x.mul(y);
  }

  backward(grad_output: tf.Tensor) {
    // o = x * y
    // dx/do = y -> treat y as constant (partial derivative)
    return [
      this.needs_input_grad[0] ? this.y.mul(grad_output) : undefined,
      this.needs_input_grad[1] ? this.x.mul(grad_output) : undefined,
    ];
  }
}

class Reshape extends Fn {
  input_shape!: number[];

  forward([x]: tf.Tensor[], { shape }: { shape: number[] }) {
    this.input_shape = shape!;
    return x.reshape(this.input_shape);
  }

  backward(grad_output: tf.Tensor) {
    let out = grad_output.reshape(this.input_shape);
    return out;
  }
}

class Sum extends Fn {
  forward([x]: tf.Tensor[], { axis }: { axis?: number | number[] }) { }
}
