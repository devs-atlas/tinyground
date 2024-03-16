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
  backward(_: any, ...__: any): any {}

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
