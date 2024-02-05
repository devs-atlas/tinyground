import nj from "@d4c/numjs";

class Tensor {
  grad?: Tensor;
  data: nj.NdArray;
  shape: number[];
  requires_grad: boolean;
  context?: Fn;

  constructor(data: number | nj.NdArray, requires_grad: boolean) {
    if (data instanceof nj.NdArray) {
      this.data = data;
      this.shape = data.shape;
    } else {
      this.data = nj.array([data]);
      this.shape = [];
    }
    this.requires_grad = requires_grad;
  }

  add(tensor: Tensor) {
    return Add.run_op(this, tensor);
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

  constructor(...tensors: Tensor[]) {
    this.needs_input_grad = tensors.map((t) => t.requires_grad);
    this.requires_grad = this.needs_input_grad.some(Boolean);
    if (this.requires_grad) {
      this.parents = tensors;
    }
  }

  backward(grad_output: nj.NdArray): (nj.NdArray | undefined)[] {
    throw new Error(
      `NotImplemented: backward not implemented for type ${typeof this}`
    );
  }

  forward(...args: nj.NdArray[]): nj.NdArray {
    throw new Error(
      `NotImplemented: forward not implemented for type ${typeof this}`
    );
  }

  // TODO: missing kwargs
  static run_op(...tensors: Tensor[]): Tensor {
    const context = new this(...tensors);
    const tensor = new Tensor(
      context.forward(...tensors.map((t) => t.data)),
      context.requires_grad
    );
    if (context.requires_grad) {
      tensor.context = context;
    }
    return tensor;
  }
}

class Add extends Fn {
  forward(x: nj.NdArray, y: nj.NdArray) {
    return x.add(y);
  }
  backward(grad_output: nj.NdArray) {
    return [
      this.needs_input_grad[0] ? grad_output : undefined,
      this.needs_input_grad[1] ? grad_output : undefined,
    ];
  }
}

function testTensors() {
  let t = new Tensor(nj.array([1, 2, 3, 4, 5]), true);

  console.log(t.toString());

  t = new Tensor(
    nj.array([
      [1, 2],
      [3, 4],
      [5, 7],
    ]),
    false
  );

  let t1 = new Tensor(
    nj.array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]),
    true
  );

  const t3 = t1.add(t);
  console.log("added:");
  console.log(t3);
}

function main() {
  testTensors();
}

main();
