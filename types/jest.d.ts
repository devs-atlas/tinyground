declare global {
  namespace jest {
    interface Matchers<R> {
      toEqualData(expected: Tensor): R;
    }
  }
}

export {}
