"use client";

import ReactFlow, {
  Background,
  Controls,
  Panel,
  ReactFlowProvider,
} from "reactflow";
import { shallow } from "zustand/shallow";

import { selector, useStore } from "../store/store";
import TensorNode from "./nodes/tensor";
import OperationNode from "./nodes/operation";

import "reactflow/dist/style.css";

const nodeTypes = { TensorNode, OperationNode };

function Flowski() {
  const store = useStore(selector, shallow);

  console.log(store);

  return (
    <ReactFlowProvider>
      <ReactFlow
        nodes={store.nodes}
        edges={store.edges}
        onNodesChange={store.onNodesChange}
        onEdgesChange={store.onEdgesChange}
        onConnect={store.onConnect}
        nodeTypes={nodeTypes}
        fitView
      >
        <Background />
        <Controls showInteractive={false} />
        <Panel position="top-left">tinyground</Panel>
      </ReactFlow>
    </ReactFlowProvider>
  );
}

export default Flowski;
