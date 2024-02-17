// UnaryOps converted to type
export type UnaryOps =
  | 'NOOP'
  | 'EXP2'
  | 'LOG2'
  | 'CAST'
  | 'SIN'
  | 'SQRT'
  | 'RECIP'
  | 'NEG';

// BinaryOps converted to type
export type BinaryOps =
  | 'ADD'
  | 'SUB'
  | 'MUL'
  | 'DIV'
  | 'MAX'
  | 'MOD'
  | 'CMPLT';

// ReduceOps converted to type
export type ReduceOps =
  | 'SUM'
  | 'MAX';

// TernaryOps converted to type
export type TernaryOps =
  | 'MULACC'
  | 'WHERE';

// MovementOps converted to type
export type MovementOps =
  | 'RESHAPE'
  | 'PERMUTE'
  | 'EXPAND'
  | 'PAD'
  | 'SHRINK'
  | 'STRIDE';

// LoadOps converted to type
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

