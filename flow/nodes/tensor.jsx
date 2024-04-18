import { Handle, NodeProps, Position } from "reactflow";

const Tensor = ({ id, data }) => {
  <>
    <input defaultValue={data.label} />

    <Handle type="target" position={Position.Top} />
    <Handle type="source" position={Position.Bottom} />
  </>;
};

export default Tensor;
