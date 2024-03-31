import LazyBuffer from "./lazy";
import Tensor from "./tensor";

export default class Fn {
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
  // @ts-ignore
  backward(_: any, ...__: any): LazyBuffer | (LazyBuffer | undefined)[] {}

  static run_op(tensors: Tensor[], options = {}): Tensor {
    const context = new this(tensors);
    const tensor = new Tensor(
      context.forward(
        tensors.map((t) => t.data),
        options
      ),
      context.requires_grad
    );
    console.log(`returning tensor of shape ${tensor.shape}`)
    if (context.requires_grad) {
      tensor.context = context;
    }
    return tensor;
  }
}
