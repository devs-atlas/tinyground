import { addEdge, applyEdgeChanges, applyNodeChanges } from "reactflow";
import { create } from "zustand";
import TensorNode from "../flow/nodes/tensor";
import OperationNode from "../flow/nodes/operation";
import Tensor from "../lib/tensor";

const useStore = create((set, get) => ({
  nodes: [
    {
      id: "a",
      type: "TensorNode",
      data: { tensor: new Tensor(5) },
      position: { x: 0, y: 0 },
    },
    {
      id: "b",
      type: "TensorNode",
      data: { tensor: new Tensor(4) },
      position: { x: -100, y: 100 },
    },
    {
      id: "c",
      type: "OperationNode",
      data: { op: "MULACC" },
      position: { x: 100, y: 100 },
    },
  ],
  edges: [],
  // edges: [{ id: "e1", source: "a", target: "b" }],
  onNodesChange: (changes) => {
    set({ nodes: applyNodeChanges(changes, get().nodes) });
  },
  onEdgesChange: (changes) => {
    set({ edges: applyEdgeChanges(changes, get().edges) });
  },
  onConnect: (connection) => {
    set({ edges: addEdge(connection, get().edges) });
  },
  addNode: (node) => {
    set({ nodes: [...get().nodes, node] });
  },
  setNodes: (nodes) => {
    set({ nodes });
  },
  setEdges: (edges) => {
    set({ edges });
  },
  nodeTypes: () => ({ TensorNode, OperationNode }),
}));

const selector = (state) => ({
  nodes: state.nodes,
  edges: state.edges,
  onNodesChange: state.onNodesChange,
  onEdgesChange: state.onEdgesChange,
  onConnect: state.onConnect,
  setNodes: state.setNodes,
  addNode: state.addNode,
  setEdges: state.setEdges,
});

export { selector, useStore };
