import { Handle, Position } from "reactflow";
// import { useLayoutEffect, useRef } from "react";

import "./tensor.css";

// show shape, requires_grad
const TensorNode = ({ data }) => {
  const { tensor } = data;
  // const inputRef = useRef();
  // NOTE: dynamic width as text updates
  // useLayoutEffect(() => {
  //   if (inputRef.current) {
  //     inputRef.current.style.width = `${Math.max(
  //       tensor.shape.length,
  //       "requires_grad: false".length
  //     )}px`;
  //   }
  // }, [tensor.shape.length]);

  return (
    <>
      <div>Requires grad: {tensor.requires_grad.toString()}</div>
      <div>Shape: {tensor.shape}</div>
      <Handle type="source" position={Position.Right} />
    </>
  );
};

export default TensorNode;
