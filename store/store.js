import { applyNodeChanges, applyEdgeChanges, addEdge } from "reactflow";
import TensorNode from "../flow/nodes/tensor";
import Tensor from "../lib/tensor";
import { create } from "zustand";
import { v4 as uuidv4 } from "uuid";

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
      position: { x: 0, y: 10 },
    },
    {
      id: "c",
      type: "OperationNode",
      data: { tensor: new Tensor(4) },
      position: { x: 0, y: 10 },
    },
  ],
  // edges: [{ id: "e1", source: "a", target: "b" }],
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
    set({
      edges: addEdge({ ...connection, id: uuidv4() }, get().edges),
    });
  },
  nodeTypes: () => ({ TensorNode }),
}));

const selector = (state) => ({
  nodes: state.nodes,
  edges: state.edges,
  onNodesChange: state.onNodesChange,
  onEdgesChange: state.onEdgesChange,
  onConnect: state.onConnect,
});

export { useStore, selector };
