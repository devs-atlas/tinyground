// operations.ts
export enum UnaryOps {
  NOOP,
  EXP2,
  LOG2,
  CAST,
  SIN,
  SQRT,
  RECIP,
  NEG,
}

export enum BinaryOps {
  ADD,
  SUB,
  MUL,
  DIV,
  MAX,
  MOD,
  CMPLT,
}

export enum ReduceOps {
  SUM,
  MAX,
}

export enum TernaryOps {
  MULACC,
  WHERE,
}

export enum MovementOps {
  RESHAPE,
  PERMUTE,
  EXPAND,
  PAD,
  SHRINK,
  STRIDE,
}

export enum LoadOps {
  // EMPTY,
  RAND,
  CONST,
  FROM,
  CONTIGUOUS,
  CUSTOM,
}

