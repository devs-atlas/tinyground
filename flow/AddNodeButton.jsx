"use client";

import { v4 as uuidv4 } from "uuid";
import Tensor from "../lib/tensor";
import { selector, useStore } from "../store/store";
import { shallow } from "zustand/shallow";

// TODO: ask nick about both UI & workflow (what makes sense from ML context?)
// TODO: abstract this away
const AddNodeButton = ({ type }) => {
  const store = useStore(selector, shallow);
  const createNode = (nodeType, ...rest) => {
    let node;
    switch (nodeType) {
      case "TensorNode":
        node = {
          id: uuidv4(),
          type: "TensorNode",
          data: { tensor: new Tensor(45) },
          position: { x: -100, y: 75 },
        };
        break;
      case "OperationNode":
        node = {
          id: uuidv4(),
          type: "OperationNode",
          data: { op: "SUM" },
          position: { x: 175, y: 200 },
        };
        break;
    }
    store.addNode(node);
  };

  return (
    <button onClick={() => createNode(type)} className="border border-black">
      Create {type}
    </button>
  );
};

export default AddNodeButton;
