export type UnaryOps =
  | 'NOOP'
  | 'EXP2'
  | 'LOG2'
  | 'CAST'
  | 'SIN'
  | 'SQRT'
  | 'RECIP'
  | 'NEG';

export type BinaryOps =
  | 'ADD'
  | 'SUB'
  | 'MUL'
  | 'DIV'
  | 'MAX'
  | 'MOD'
  | 'CMPLT';

export type ReduceOps =
  | 'SUM'
  | 'MAX';

export type TernaryOps =
  | 'MULACC'
  | 'WHERE';

export type MovementOps =
  | 'RESHAPE'
  | 'PERMUTE'
  | 'EXPAND'
  | 'PAD'
  | 'SHRINK'
  | 'STRIDE';

export type LoadOps =
  | 'EMPTY'
  | 'RAND'
  | 'CONST'
  | 'FROM'
  | 'CONTIGUOUS'
  | 'CUSTOM';

// Op converted to type
export type Op =
  | UnaryOps
  | BinaryOps
  | ReduceOps
  | TernaryOps
  | MovementOps
  | LoadOps;

