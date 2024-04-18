"use client";

import ReactFlow, {
  ReactFlowProvider,
  Background,
  Controls,
  Panel,
} from "reactflow";
import { shallow } from "zustand/shallow";

import { useStore, selector } from "../store/store";

import "reactflow/dist/style.css";

function Flowski() {
  const store = useStore(selector, shallow);

  return (
    <ReactFlowProvider>
      <ReactFlow
        nodes={store.nodes}
        edges={store.edges}
        onNodesChange={store.onNodesChange}
        onEdgesChange={store.onEdgesChange}
        onConnect={store.onConnect}
        nodeTypes={store.nodeTypes}
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
