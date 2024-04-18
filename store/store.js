import { applyNodeChanges, applyEdgeChanges, addEdge } from "reactflow";
import Tensor from "../flow/nodes/tensor";
import { create } from "zustand";
import { v4 as uuidv4 } from "uuid";

const useStore = create((set, get) => ({
  nodes: [
    {
      id: "a",
      type: "tensor",
      data: { label: "TinyGround" },
      position: { x: 0, y: 0 },
    },
    {
      id: "b",
      type: "tensor",
      data: { label: "TinyGround" },
      position: { x: 0, y: 10 },
    },
  ],
  edges: [{ id: "e1", source: "a", target: "b" }],
  onNodesChange: (changes) => {
    set({
      nodes: applyNodeChanges(changes, get().nodes),
    });
  },
  onEdgesChange: (changes) => {
    set({
      edges: applyEdgeChanges(changes, get().edges),
    });
  },
  onConnect: (connection) => {
    console.log("called");
    set({
      // TODO: probably don't want uuid but
      edges: addEdge({ ...connection, id: uuidv4() }, get().edges),
    });
  },
  nodeTypes: () => ({ tensor: Tensor }),
}));

const selector = (state) => ({
  nodes: state.nodes,
  edges: state.edges,
  onNodesChange: state.onNodesChange,
  onEdgesChange: state.onEdgesChange,
  onConnect: state.onConnect,
  nodeTypes: state.nodeTypes,
});

export { useStore, selector };
