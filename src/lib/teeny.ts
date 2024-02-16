import { NdArray as NdA, default as nj } from "@d4c/numjs";

export class Tensor {
  grad?: Tensor;
  data: NdA;
  shape: number[];
  requires_grad: boolean;
  context?: Fn;

  constructor(data: number | NdA, requires_grad: boolean) {
    if (data instanceof NdA) {
      this.data = data;
      this.shape = data.shape;
    } else {
      this.data = nj.array([data]);
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

  // reduce(fn: Fn, axis?: number | number[], keepdim = false): Tensor {
  //   let axis_: number[];
  //
  //   if (axis === undefined) {
  //     axis_ = Array.from(Array(this.shape.length).keys());
  //   } else if (typeof axis === "number") {
  //     axis_ = [axis];
  //   } else {
  //     axis_ = axis;
  //   }
  //
  //   for (let i = 0; i < axis_.length; ++i) {
  //     if (axis_[i] < 0) {
  //       axis_[i] += this.shape.length;
  //     }
  //   }
  //   const shape = this.shape.filter((_, i) => !axis_.includes(i));
  //
  //   if (this.shape.includes(0) && !shape.includes(0)) {
  //     // TODO:
  //     return;
  //   }
  //
  //   const new_shape = this.shape.filter((s, i) => (axis_.includes(i) ? 1 : s));
  //   const tensor = Fn.run_op([this], { new_shape });
  // }

  // prob shouldnt be static
  // static _sum_along_axis(tensor: Tensor, axis?: number | number[]) :  Tensor {
  //   shape = tensor.shape
  //
  //   // dont do the call like this fr
  //   if !axis:
  //     return tensor.data.sum()
  //
  //   for(let i=0; i < )
  // }

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

  forward(_: any, ...__: any): any {}
  backward(_: any, ...__: any): any {}

  static run_op(tensors: Tensor[], options = {}): Tensor {
    const context = new this(tensors);
    const tensor = new Tensor(
      context.forward(
        tensors.map((t) => t.data),
        options
      ),
      context.requires_grad
    );
    if (context.requires_grad) {
      tensor.context = context;
    }
    return tensor;
  }
}

// class Expand extends Fn {
//   input_shape: Nda;
//   forward([x]: NdA[], new_shape) {
//     this.input_shape = x.shape;
//     return broadcast_to(x, new_shape);
//   }
//   backward( grad_output: NdA ) {
//     return grad_output.sum()
//   }
// }

class Add extends Fn {
  forward([x, y]: NdA[]) {
    return x.add(y);
  }
  backward(grad_output: NdA) {
    // o = x + y
    // dx/do = 1 -> multiply by grad_output to represent dx/dL
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output : undefined,
    ];
  }
}

class Sub extends Fn {
  forward([x, y]: NdA[]) {
    return x.subtract(y);
  }
  backward(grad_output: NdA) {
    // o = x - y = x - 1y
    // dy/do = -1
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output.negative() : undefined,
    ];
  }
}

class Mul extends Fn {
  x!: NdA;
  y!: NdA;

  forward([x, y]: NdA[]) {
    this.x = x;
    this.y = y;
    return x.multiply(y);
  }

  backward(grad_output: NdA) {
    // o = x * y
    // dx/do = y -> treat y as constant (partial derivative)
    return [
      this.needs_input_grad[0] ? this.y.multiply(grad_output) : undefined,
      this.needs_input_grad[1] ? this.x.multiply(grad_output) : undefined,
    ];
  }
}

class Reshape extends Fn {
  input_shape!: number[];

  forward([x]: NdA[], { shape }: { shape: number[] }) {
    this.input_shape = shape!;
    return x.reshape(...this.input_shape);
  }

  backward(grad_output: NdA) {
    let out = grad_output.reshape(this.input_shape);
    return out;
  }
}

class Expand extends Fn {
  input_shape!: number[];
  forward([x]: NdA[], { shape }: { shape: number[] }) {
    this.input_shape = shape!;
    return x;
  }
}

// class Sum extends Fn {
//   forward(tensor: Tensor, )
// }
