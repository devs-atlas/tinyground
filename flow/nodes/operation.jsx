import "./operation.css";

import { Handle, Position, PanelPosition } from "reactflow";
import {
  BinaryOps,
  MovementOps,
  ReduceOps,
  TernaryOps,
  UnaryOps,
} from "../../lib/mlops";

// inputs; depend on type (unary, binary, ternary - only those)
// opname
// one output
// general op type broken down, see https://github.com/tinygrad/teenygrad/blob/main/teenygrad/ops.py

const getOpInfo = (opName) => {
  if (UnaryOps[opName] !== undefined) return ["Unary", 1];
  if (BinaryOps[opName] !== undefined) return ["Binary", 2];
  if (ReduceOps[opName] !== undefined) return ["Reduce", 1];
  if (TernaryOps[opName] !== undefined) return ["Ternary", 3];
  if (MovementOps[opName] !== undefined) return ["Movement", 1];
};

const OperationNode = ({ data }) => {
  // TODO: memoize?
  const { op } = data;
  const [type, inputCount] = getOpInfo(op);

  return (
    <>
      <div>Name: {op}</div>
      <div>Type: {type}</div>
      {/* TODO: The right number of handles is showing up, but the 
      styling does absolutely nothing - figure out a connections workflow*/}
      {/* nic wants n visually separate handles */}
      {Array.from({ length: inputCount }).map((_, i) => (
        <Handle key={i} type="target" position={Position.Left} />
      ))}
    </>
  );
};

export default OperationNode;
