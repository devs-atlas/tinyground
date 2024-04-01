import LazyBuffer from "./lazy";

import Tensor from "./tensor";

export default class Fn {
  needs_input_grad;
  requires_grad;
  parents;

  constructor(tensors) {
    this.needs_input_grad = tensors.map((t) => t.requires_grad);
    this.requires_grad = this.needs_input_grad.some(Boolean);
    if (this.requires_grad) {
      this.parents = tensors;
    }
  }

  forward(_, ...__) {}

  // @ts-ignore
  backward(_, ...__) {}

  static run_op(tensors, options = {}) {
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
