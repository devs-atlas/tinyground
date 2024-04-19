import "./operation.css";

import { Handle, Position } from "reactflow";
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

// TODO: styling does nothing here
const getHandlePositions = (count) => {
  switch (count) {
    case 1:
      return [{ position: Position.Left }];
    case 2:
      return [
        { position: Position.Left, top: "30%" },
        { position: Position.Left, top: "70%" },
      ];
    case 3:
      return [
        { position: Position.Left, top: "20%" },
        { position: Position.Left, top: "50%" },
        { position: Position.Left, top: "80%" },
      ];
    default:
      return [];
  }
};

const OperationNode = ({ data }) => {
  // TODO: memoize?
  const { op } = data;
  const [type, inputCount] = getOpInfo(op);
  const handles = getHandlePositions(inputCount);

  return (
    <>
      <div>Name: {op}</div>
      <div>Type: {type}</div>
      {/* TODO: The right number of handles is showing up, but the 
      styling does absolutely nothing - figure out a connections workflow*/}
      {/* nic wants n visually separate handles */}
      {handles.map((handle, i) => (
        <Handle key={i} type="source" position={handle.position} />
      ))}
    </>
  );
};

export default OperationNode;
