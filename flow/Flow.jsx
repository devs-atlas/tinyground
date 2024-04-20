"use client";

import ReactFlow, {
  Background,
  Controls,
  Panel,
  ReactFlowProvider,
} from "reactflow";
import { shallow } from "zustand/shallow";

import { selector, useStore } from "../store/store";
import AddNodeButton from "./AddNodeButton";
import OperationNode from "./nodes/operation";
import TensorNode from "./nodes/tensor";

import "reactflow/dist/style.css";
import "./Flow.css";

const nodeTypes = { TensorNode, OperationNode };

function Flow() {
  const store = useStore(selector, shallow);

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
        <Panel position="top-left" className="border border-black">
          tinyground
        </Panel>
        <Panel position="top-right">
          <AddNodeButton type={"TensorNode"} />
          <AddNodeButton type={"OperationNode"} />
        </Panel>
      </ReactFlow>
    </ReactFlowProvider>
  );
}

export default Flow;
