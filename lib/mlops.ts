import Fn from "./fn";
import LazyBuffer from "./lazy";
import { argsort } from "./utils";

export class Contiguous extends Fn {
  forward([x]: LazyBuffer[]) {
    return x.contiguous();
  }
  backward(grad_output: LazyBuffer) {
    return grad_output;
  }
}

export class ContiguousBackward extends Fn {
  forward([x]: LazyBuffer[]) {
    return x;
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.contiguous();
  }
}

export class Cast extends Fn {
  input_dtype!: string;

  //TODO: maybe add dtype type thing
  forward([x]: LazyBuffer[], { dtype }: { dtype: string }) {
    this.input_dtype = x.dtype;
    return x.cast(dtype);
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.cast(this.input_dtype);
  }
}

// UNARY OPS

export class Zero extends Fn {
  forward([x]: LazyBuffer[]) {
    return x.const(0);
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.const(0);
  }
}

export class Neg extends Fn {
  forward([x]: LazyBuffer[]) {
    return x.e("NEG");
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.e("NEG");
  }
}

export class Sin extends Fn {
  x!: LazyBuffer;
  forward([x]: LazyBuffer[]) {
    this.x = x;
    return x.e("SIN");
  }
  backward(grad_output: LazyBuffer) {
    return this.x
      .const(Math.PI / 2)
      .e("SUB", this.x)
      .e("SIN")
      .e("MUL", grad_output);
  }
}

export class Relu extends Fn {
  ret!: LazyBuffer;
  forward([x]: LazyBuffer[]) {
    this.ret = x.e("MAX", x.const(0));
    return this.ret;
  }
  backward(grad_output: LazyBuffer) {
    return this.ret.const(0).e("CMPLT", this.ret).e("MUL", grad_output);
  }
}

export class Log extends Fn {
  x!: LazyBuffer;
  forward([x]: LazyBuffer[]) {
    this.x = x;
    //TODO: is it dumb to do this - dividing and multiplying by log2 instead of just using default
    //TODO: is Math.log(2) == Math.LOG2E?
    return x.e("LOG2").e("MUL", x.const(Math.log(2)));
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.e("DIV", this.x);
  }
}

export class Exp extends Fn {
  ret!: LazyBuffer;
  forward([x]: LazyBuffer[]) {
    this.ret = x.e("MUL", x.const(1 / Math.log(2))).e("EXP2");
    return this.ret;
  }
  backward(grad_output: LazyBuffer) {
    return this.ret.e("MUL", grad_output);
  }
}

export class Sqrt extends Fn {
  ret!: LazyBuffer;
  forward([x]: LazyBuffer[]) {
    this.ret = x.e("SQRT");
    return this.ret;
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.e("DIV", this.ret.e("MUL", this.ret.const(2)));
  }
}

export class Sigmoid extends Fn {
  ret!: LazyBuffer;
  forward([x]: LazyBuffer[]) {
    this.ret = x
      .const(1)
      .e("DIV", x.const(1).e("ADD", x.e("MUL", x.const(-1 / Math.log(2)))))
      .e("EXP2");
  }
}

// BINARY OPS

export class Less extends Fn {
  forward([x, y]: LazyBuffer[]) {
    return x.e("CMPLT", y);
  }
  backward(grad_output: LazyBuffer) {
    return [undefined, undefined];
  }
}

// TODO: change other ops methods
export class Add extends Fn {
  forward([x, y]: LazyBuffer[]) {
    return x.e("ADD", y);
  }
  backward(grad_output: LazyBuffer) {
    //TODO: Do I return as list?
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output : undefined,
    ];
  }
}

export class Sub extends Fn {
  forward([x, y]: LazyBuffer[]) {
    return x.e("SUB", y);
  }
  backward(grad_output: LazyBuffer) {
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output.e("NEG") : undefined,
    ];
  }
}

export class Mul extends Fn {
  x!: LazyBuffer;
  y!: LazyBuffer;

  forward([x, y]: LazyBuffer[]) {
    this.x = x;
    this.y = y;
    return x.e("MUL", y);
  }

  backward(grad_output: LazyBuffer) {
    return [
      this.needs_input_grad[0] ? this.y.e("MUL", grad_output) : undefined,
      this.needs_input_grad[1] ? this.x.e("MUL", grad_output) : undefined,
    ];
  }
}

export class Div extends Fn {
  x!: LazyBuffer;
  y!: LazyBuffer;

  forward([x, y]: LazyBuffer[]) {
    this.x = x;
    this.y = y;
    return x.e("DIV", y);
  }

  backward(grad_output: LazyBuffer) {
    return [
      this.needs_input_grad[0] ? grad_output.e("DIV", this.y) : undefined,
      this.needs_input_grad[1]
        ? grad_output
            .e("NEG")
            .e("MUL", this.x)
            .e("DIV", this.y.e("MUL", this.y))
        : undefined,
    ];
  }
}

// TERNARY OPS

//TODO: ADD SUPPORT FOR WHERE
// export class Where extends Fn {
//   x!: LazyBuffer;
//   forward(x: LazyBuffer, y: LazyBuffer, z: LazyBuffer) {
//     this.x = x;
//     return x.e("WHERE", y, z);
//   }
//   backward(grad_output: LazyBuffer) {
//     return [
//       undefined,
//       this.needs_input_grad[1]
//         ? this.x.e("WHERE", grad_output, grad_output.const(0))
//         : undefined,
//       this.needs_input_grad[2]
//         ? this.x.e("WHERE", grad_output.const(0), grad_output)
//         : undefined,
//     ];
//   }
// }

// REDUCE OPS

export class Sum extends Fn {
  input_shape!: number[];
  forward([x]: LazyBuffer[], { new_shape }: { new_shape: number[] }) {
    this.input_shape = x.shape;
    return x.r("SUM", new_shape);
  }
  backward(grad_output: LazyBuffer) {
    return grad_output.expand(this.input_shape);
  }
}

export class Max extends Fn {
  x!: LazyBuffer;
  ret!: LazyBuffer;
  forward([x]: LazyBuffer[], { new_shape }: { new_shape: number[] }) {
    this.x = x;
    this.ret = x.r("MAX", new_shape);
    return this.ret;
  }
  backward(grad_output: LazyBuffer) {
    let max_is_1s = this.x
      .const(1)
      .e("SUB", this.x.e("CMPLT", this.ret.expand(this.x.shape)));
    let div = max_is_1s.r("SUM", grad_output.shape).expand(this.x.shape);
    return max_is_1s.e("DIV", div).e("MUL", grad_output.expand(this.x.shape));
  }
}

// MOVEMENT OPS

export class Expand extends Fn {
  input_shape!: number[];

  forward([x]: LazyBuffer[], { shape }: { shape: number[] }) {
    this.input_shape = x.shape;
    return x.expand(shape);
  }

  backward(grad_output: LazyBuffer) {
    return grad_output.r("SUM", this.input_shape);
  }
}

export class Reshape extends Fn {
  input_shape!: number[];

  forward([x]: LazyBuffer[], { shape }: { shape: number[] }) {
    this.input_shape = x.shape;
    return x.reshape(shape);
  }

  backward(grad_output: LazyBuffer) {
    return grad_output.reshape(this.input_shape);
  }
}

export class Permute extends Fn {
  input_order!: number[];

  forward([x]: LazyBuffer[], { order }: { order: number[] }) {
    this.input_order = order;
    return x.permute(order);
  }

  backward(grad_output: LazyBuffer) {
    let inverseOrder = argsort(this.input_order);
    return grad_output.permute(inverseOrder);
  }
}

export class Pad extends Fn {
  narg!: Array<[number, number]>;

  forward([x]: LazyBuffer[], { arg }: { arg: Array<[number, number]> }) {
    this.narg = arg.map((p, i) => [p[0], x.shape[i] + p[0]]);
    return x.pad(arg);
  }

  backward(grad_output: LazyBuffer) {
    return grad_output.shrink(this.narg);
  }
}

export class Shrink extends Fn {
  narg!: Array<[number, number]>;

  forward([x]: LazyBuffer[], { arg }: { arg: Array<[number, number]> }) {
    this.narg = arg.map((p, i) => [p[0], x.shape[i] - p[1]]);
    return x.shrink(arg);
  }

  backward(grad_output: LazyBuffer) {
    // Ensure all elements in narg are numbers; otherwise, handle error or exception.
    return grad_output.pad(this.narg as Array<[number, number]>);
  }
}

export class Flip extends Fn {
  arg!: number[];

  forward([x]: LazyBuffer[], { axis }: { axis: number[] }) {
    this.arg = x.shape.map((_, i) => (axis.includes(i) ? -1 : 1));
    return x.stride(this.arg);
  }

  backward(grad_output: LazyBuffer) {
    return grad_output.stride(this.arg);
  }
}
