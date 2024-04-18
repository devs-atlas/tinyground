import Fn from "./fn";
import { argsort } from "./utils";

export class Contiguous extends Fn {
  forward([x]) {
    return x.contiguous();
  }
  backward(grad_output) {
    return [grad_output];
  }
}

export class ContiguousBackward extends Fn {
  forward([x]) {
    return x;
  }
  backward(grad_output) {
    return [grad_output.contiguous()];
  }
}

export class Cast extends Fn {
  //TODO: maybe add dtype type thing
  forward([x], { dtype }) {
    this.input_dtype = x.dtype;
    return x.cast(dtype);
  }
  backward(grad_output) {
    return [grad_output.cast(this.input_dtype)];
  }
}

// UNARY OPS

export class Zero extends Fn {
  forward([x]) {
    return x.const(0);
  }
  backward(grad_output) {
    return [grad_output.const(0)];
  }
}

export class Neg extends Fn {
  forward([x]) {
    return x.e("NEG");
  }
  backward(grad_output) {
    return [grad_output.e("NEG")];
  }
}

export class Sin extends Fn {
  forward([x]) {
    this.x = x;
    return x.e("SIN");
  }
  backward(grad_output) {
    return [
      this.x
        .const(Math.PI / 2)
        .e("SUB", this.x)
        .e("SIN")
        .e("MUL", grad_output),
    ];
  }
}

export class Relu extends Fn {
  forward([x]) {
    this.ret = x.e("MAX", x.const(0));
    return this.ret;
  }
  backward(grad_output) {
    return [this.ret.const(0).e("CMPLT", this.ret).e("MUL", grad_output)];
  }
}

export class Log extends Fn {
  forward([x]) {
    this.x = x;
    //TODO: is it dumb to do this - dividing and multiplying by log2 instead of just using default
    //TODO: is Math.log(2) == Math.LOG2E?
    return x.e("LOG2").e("MUL", x.const(Math.log(2)));
  }
  backward(grad_output) {
    return [grad_output.e("DIV", this.x)];
  }
}

export class Exp extends Fn {
  forward([x]) {
    this.ret = x.e("MUL", x.const(1 / Math.log(2))).e("EXP2");
    return this.ret;
  }
  backward(grad_output) {
    return [this.ret.e("MUL", grad_output)];
  }
}

export class Sqrt extends Fn {
  forward([x]) {
    this.ret = x.e("SQRT");
    return this.ret;
  }
  backward(grad_output) {
    return [grad_output.e("DIV", this.ret.e("MUL", this.ret.const(2)))];
  }
}

export class Sigmoid extends Fn {
  forward([x]) {
    this.ret = [
      x
        .const(1)
        .e("DIV", x.const(1).e("ADD", x.e("MUL", x.const(-1 / Math.log(2)))))
        .e("EXP2"),
    ];
    return this.ret;
  }
  // TODO: backward
}

// BINARY OPS

export class Less extends Fn {
  forward([x, y]) {
    return x.e("CMPLT", y);
  }
  backward(grad_output) {
    return [undefined, undefined];
  }
}

// TODO: change other ops methods
export class Add extends Fn {
  forward([x, y]) {
    return x.e("ADD", y);
  }
  backward(grad_output) {
    //TODO: Do I return as list?
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output : undefined,
    ];
  }
}

export class Sub extends Fn {
  forward([x, y]) {
    return x.e("SUB", y);
  }
  backward(grad_output) {
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output.e("NEG") : undefined,
    ];
  }
}

export class Mul extends Fn {
  forward([x, y]) {
    this.x = x;
    this.y = y;
    return x.e("MUL", y);
  }

  backward(grad_output) {
    return [
      this.needs_input_grad[0] ? this.y.e("MUL", grad_output) : undefined,
      this.needs_input_grad[1] ? this.x.e("MUL", grad_output) : undefined,
    ];
  }
}

export class Div extends Fn {
  forward([x, y]) {
    this.x = x;
    this.y = y;
    return x.e("DIV", y);
  }

  backward(grad_output) {
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
//   x;
//   forward(x, y, z) {
//     this.x = x;
//     return x.e("WHERE", y, z);
//   }
//   backward(grad_output) {
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
  forward([x], { new_shape }) {
    this.input_shape = x.shape;
    return x.r("SUM", new_shape);
  }
  backward(grad_output) {
    return [grad_output.expand(this.input_shape)];
  }
}

export class Max extends Fn {
  forward([x], { new_shape }) {
    this.x = x;
    this.ret = x.r("MAX", new_shape);
    return this.ret;
  }
  backward(grad_output) {
    let max_is_1s = this.x
      .const(1)
      .e("SUB", this.x.e("CMPLT", this.ret.expand(this.x.shape)));
    let div = max_is_1s.r("SUM", grad_output.shape).expand(this.x.shape);
    return [max_is_1s.e("DIV", div).e("MUL", grad_output.expand(this.x.shape))];
  }
}

// MOVEMENT OPS

export class Expand extends Fn {
  forward([x], { shape }) {
    this.input_shape = x.shape;
    return x.expand(shape);
  }

  backward(grad_output) {
    return [grad_output.r("SUM", this.input_shape)];
  }
}

export class Reshape extends Fn {
  forward([x], { shape }) {
    this.input_shape = x.shape;
    return x.reshape(shape);
  }

  backward(grad_output) {
    return [grad_output.reshape(this.input_shape)];
  }
}

export class Permute extends Fn {
  forward([x], { order }) {
    this.input_order = order;
    return x.permute(order);
  }

  backward(grad_output) {
    let inverseOrder = argsort(this.input_order);
    return [grad_output.permute(inverseOrder)];
  }
}

export class Pad extends Fn {
  forward([x], { arg }) {
    this.narg = arg.map((p, i) => [p[0], x.shape[i] + p[0]]);
    return x.pad(arg);
  }

  backward(grad_output) {
    return [grad_output.shrink(this.narg)];
  }
}

export class Shrink extends Fn {
  forward([x], { arg }) {
    this.narg = arg.map((p, i) => [p[0], x.shape[i] - p[1]]);
    return x.shrink(arg);
  }

  backward(grad_output) {
    // Ensure all elements in narg are numbers; otherwise, handle error or exception.
    return [grad_output.pad(this.narg)];
  }
}

export class Flip extends Fn {
  forward([x], { axis }) {
    this.arg = x.shape.map((_, i) => (axis.includes(i) ? -1 : 1));
    return x.stride(this.arg);
  }

  backward(grad_output) {
    return [grad_output.stride(this.arg)];
  }
}

const UnaryOps = Object.freeze({
  NOOP: 0,
  EXP2: 1,
  LOG2: 2,
  CAST: 3,
  SIN: 4,
  SQRT: 5,
  RECIP: 6,
  NEG: 7,
});

const BinaryOps = Object.freeze({
  ADD: 0,
  SUB: 1,
  MUL: 2,
  DIV: 3,
  MAX: 4,
  MOD: 5,
  CMPLT: 6,
});

const ReduceOps = Object.freeze({
  SUM: 0,
  MAX: 1,
});

const TernaryOps = Object.freeze({
  MULACC: 0,
  WHERE: 1,
});

const MovementOps = Object.freeze({
  RESHAPE: 0,
  PERMUTE: 1,
  EXPAND: 2,
  PAD: 3,
  SHRINK: 4,
  STRIDE: 5,
});

const LoadOps = Object.freeze({
  EMPTY: 0,
  RAND: 1,
  CONST: 2,
  FROM: 3,
  CONTIGUOUS: 4,
  CUSTOM: 5,
});

const Ops = Object.freeze({
  ...UnaryOps,
  ...BinaryOps,
  ...ReduceOps,
  ...TernaryOps,
  ...MovementOps,
  ...LoadOps,
});

export {
  UnaryOps,
  BinaryOps,
  ReduceOps,
  TernaryOps,
  MovementOps,
  LoadOps,
  Ops,
};
