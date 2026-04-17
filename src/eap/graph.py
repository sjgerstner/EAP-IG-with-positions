from typing import List, Dict, Union, Tuple, Literal, Optional, Set
import json
import heapq

from einops import einsum, reduce
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig
import numpy as np

from .visualization import get_color, generate_random_color

class Node:
    """
    A node in our computational graph. The in_hook is the TL hook into its inputs, 
    while the out_hook gets its outputs.
    """
    name: str
    layer: int
    in_hook: str
    out_hook: str
    index: Tuple
    parents: Set['Node']
    parent_edges: Set['Edge']
    children: Set['Node']
    child_edges: Set['Edge']
    in_graph: bool
    score: Optional[float]
    neurons: Optional[torch.Tensor]
    neurons_scores: Optional[torch.Tensor]
    qkv_inputs: Optional[List[str]]

    def __init__(self, name: str, layer:int, in_hook: List[str], out_hook: str, index: Tuple, 
                 graph:'Graph', qkv_inputs: Optional[List[str]]=None):
        self.name = name
        self.layer = layer
        self.in_hook = in_hook
        self.out_hook = out_hook 
        self.index = index
        self.graph = graph
        self.parents = set()
        self.children = set()
        self.parent_edges = set()
        self.child_edges = set()
        self.qkv_inputs = qkv_inputs

    def __repr__(self):
        return f'Node({self.name}, in_graph: {self.in_graph})'

    def __hash__(self):
        return hash(self.name)

    # Nodes just report back their in_graph/score/neurons_in_graph/neurons_scores status from the graph
    @property
    def in_graph(self):
        return self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)]

    @in_graph.setter
    def in_graph(self, value):
        self.graph.nodes_in_graph[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def score(self):
        if self.graph.nodes_scores is None:
            return None
        return self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def score(self, value):
        if self.graph.nodes_scores is None:
            raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
        self.graph.nodes_scores[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def neurons(self):
        if self.graph.neurons is None:
            return None
        return self.graph.neurons[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def neurons(self, value):
        if self.graph.neurons is None:
            raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
        self.graph.neurons[self.graph.forward_index(self, attn_slice=False)] = value

    @property
    def neurons_scores(self):
        if self.graph.neurons_scores is None:
            return None
        return self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)]

    @score.setter
    def neurons_scores(self, value):
        if self.graph.neurons_scores is None:
            raise RuntimeError(f"Cannot set score for node {self.name} because the graph does not have node scores enabled")
        self.graph.neurons_scores[self.graph.forward_index(self, attn_slice=False)] = value

class LogitNode(Node):
    def __init__(self, n_layers:int, graph: 'Graph'):
        name = 'logits'
        index = slice(None)
        super().__init__(name, n_layers - 1, f"blocks.{n_layers - 1}.hook_resid_post", '', index, graph)

    @property
    def in_graph(self):
        return True

    @in_graph.setter
    def in_graph(self, value):
        raise ValueError(f"Cannot set in_graph for logits node (always True)")

class MLPNode(Node):
    def __init__(
        self, layer: int, graph: 'Graph',
        #pos: Optional[int] = None
    ):
        name = f'm{layer}'
        # if pos:
        #     name += f'.pos{pos}'
        index = slice(None)
        super().__init__(name, layer, f"blocks.{layer}.hook_mlp_in", f"blocks.{layer}.hook_mlp_out", index, graph)

class AttentionNode(Node):
    head: int
    def __init__(self, layer:int, head:int, graph: 'Graph'):
        name = f'a{layer}.h{head}'
        self.head = head
        index = (slice(None), slice(None), head)
        super().__init__(
            name,
            layer,
            f'blocks.{layer}.hook_attn_in',
            f"blocks.{layer}.attn.hook_result",
            index,
            graph,
            qkv_inputs=[f'blocks.{layer}.hook_{letter}_input' for letter in 'qkv']
        )

class InputNode(Node):
    def __init__(self, graph: 'Graph', name='input'):
        #name = 'input'
        index = slice(None)
        super().__init__(name, 0, '', "hook_embed", index, graph)

class Edge:
    """An Edge in a graph. 
    Attributes:
        name: (str): the edge's name, given as [PARENT]->[CHILD]<[OPTIONAL QKV>]; the latter applies only if [CHILD] is an AttentionNode
        parent: (Node): the parent node of the edge
        child: (Node): the child node of the edge
        hook: (str): the hook into the child node
        index: (Tuple): the index of the child node (only really relevant for AttentionNodes)
        score: (Optional[float]): the score of the edge (given by an attribution method)
        in_graph: (bool): whether the edge is in the graph or not"""

    name: str
    parent: Node
    child: Node
    hook: str
    index: Tuple
    graph: 'Graph'
    def __init__(
        self,
        graph: 'Graph', parent: Node, child: Node,
        qkv:Optional[Literal["q", "k", "v"]]=None,
        #pos:Optional[int]=None,
    ):
        self.graph = graph
        self.name = f'{parent.name}->{child.name}' if qkv is None else f'{parent.name}->{child.name}<{qkv}>'
        self.parent = parent
        self.child = child
        self.qkv = qkv
        self.matrix_index = (graph.forward_index(parent, attn_slice=False), graph.backward_index(child, qkv, attn_slice=False))
        # if pos is None:
        #     self.full_matrix_index = self.matrix_index
        # else:
        #     self.full_matrix_index = (pos, *self.matrix_index)

        if isinstance(child, AttentionNode):
            if qkv is None:
                raise ValueError(f'Edge({self.name}): Edges to attention heads must have a non-none value for qkv.')
            self.hook = f'blocks.{child.layer}.hook_{qkv}_input'
            self.index = (slice(None), slice(None), child.head)
        else:
            self.index = child.index
            self.hook = child.in_hook

    def __repr__(self):
        return f'Edge({self.name}, score: {self.score}, in_graph: {self.in_graph})'

    def __hash__(self):
        return hash(self.name)

    @property
    def score(self):
        return self.graph.scores[self.matrix_index]

    @score.setter
    def score(self, value):
        self.graph.scores[self.matrix_index] = value

    @property
    def in_graph(self):
        return self.graph.in_graph[self.matrix_index]

    @in_graph.setter
    def in_graph(self, value):
        self.graph.in_graph[self.matrix_index] = value

class GraphConfig(dict):
    def __init__(self, *args, **kwargs):
        super(GraphConfig, self).__init__(*args, **kwargs)
        self.__dict__ = self

class Graph:
    """
    Represents a graph that consists of nodes and edges.

    Attributes:
        nodes (Dict[str, Node]): A dictionary of nodes in the graph, where the key is the node name and the value is the node object.
        edges (Dict[str, Edge]): A dictionary of edges in the graph, where the key is the edge name and the value is the edge object.
        n_forward (int): The number of forward nodes in the graph, i.e. the # of nodes whose output activations we care about
        n_backward (int): The number of backward nodes/indices in the graph, i.e. the # of nodes whose input gradients we care about. Note that attention heads have 3 inputs that need to be dealt with during a backward pass
        cfg (HookedTransformerConfig): The configuration object for the graph.
    """
    nodes: Dict[str, Node]  # Maps from node names ('input', 'a0.h0', 'm0', 'logits', etc.) to Node objects
    edges: Dict[str, Edge]  # Maps from edge names ('input->a0.h0', 'a0.h0->m0', etc.) to Edge objects. Attn edges are denoted as 'input->a0.h0<q>', 'input->a0.h0<k>', 'input->a0.h0<v>'
    n_forward: int  # the number of forward (source) nodes
    n_backward: int  # the number of backward (destination) nodes
    scores: torch.Tensor  # tensor of edge scores, shape (n_forward, n_backward)
    in_graph: torch.Tensor  # (n_forward, n_backward) tensor of whether the edge is in the graph
    neurons_scores: Optional[torch.Tensor]  # (n_forward, d_model) tensor of neuron scores for each forward node. If a neuron's score is NaN, this indicates it has not been scored, and needs to stay in the graph.
    neurons_in_graph: Optional[torch.Tensor]  # (n_forward, d_model) tensor of whether the neuron is in the graph
    nodes_scores: Optional[torch.Tensor]  # (n_forward) tensor of source node scores. If None, nodes have no scores. If a node's score is NaN, this indicates it has not been scored, and needs to stay in the graph.
    nodes_in_graph: torch.Tensor  # (n_forward) tensor of whether the (source) node is in the graph
    positional_scores: Optional[torch.Tensor] #(n_pos, n_forward, n_backward)
    positional_edges_in_graph: Optional[torch.Tensor] #(n_pos, n_forward, n_backward)
    #positional_nodes_in_graph: Optional[torch.Tensor] #(n_pos, n_forward)
    forward_to_backward: torch.Tensor
    real_edge_mask: torch.Tensor   # (n_forward, n_backward) tensor of whether the edge is real (some edges are not real, e.g. m10->m2)
    mlp_mask_forward: torch.Tensor # (n_forward) tensor of whether the node is an MLP node
    mlp_mask_backward: torch.Tensor #(n_backward)
    cfg: GraphConfig

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.n_forward = 0
        self.n_backward = 0

    def add_edge(
        self,
        parent:Node, child:Node,
        qkv:Optional[Literal["q", "k", "v"]]=None,
        #pos:Optional[int]=None,
    ):
        edge = Edge(
            self, parent, child, qkv,
            #pos=pos
        )
        self.real_edge_mask[edge.matrix_index] = True
        self.edges[edge.name] = edge
        parent.children.add(child)
        parent.child_edges.add(edge)
        child.parents.add(parent)
        child.parent_edges.add(edge)


    def prev_index(self, node: Node) -> Union[int, slice]:
        """Return the forward index before which all nodes contribute to the input of the given node
        Args:
            node (Node): The node to get the prev forward index of
        Returns:
            Union[int, slice]: an index representing the prev forward index of the node
        """
        if isinstance(node, InputNode):
            return 0
        elif isinstance(node, LogitNode):
            return self.n_forward
        elif isinstance(node, MLPNode):
            if self.cfg['parallel_attn_mlp']:
                return 1 + node.layer * (self.cfg['n_heads'] + 1)
            else:
                return 1 + node.layer * (self.cfg['n_heads'] + 1) + self.cfg['n_heads']
        elif isinstance(node, AttentionNode):
            i =  1 + node.layer * (self.cfg['n_heads'] + 1)
            return i
        else:
            raise ValueError(f"Invalid node: {node} of type {type(node)}")

    @classmethod
    def _n_forward(cls, cfg) -> int:
        return 1 + cfg.n_layers * (cfg.n_heads + 1)

    @classmethod
    def _n_backward(cls, cfg) -> int:
        return cfg.n_layers * (3 * cfg.n_heads + 1) + 1

    @classmethod
    def _forward_index(cls, cfg, node_name:str, attn_slice:bool=False) -> int:
        """Given a model's config and a node specification, return the forward (source) index of the node in the graph. The forward index is the index of the node in the forward pass of the model, which is used to index into the graph's tensors.
        
        Args:
            cfg (_type_): a (HookedTransformer) config object
            node_name (str): Name of the node: 'input', 'logits', 'm0', 'a0.h0', etc.
            attn_slice (bool, optional): _description_. Defaults to False.

        Returns:
            int: the forward index
        """
        if node_name == 'input':
            return 0
        elif node_name == 'logits':
            return 1 + cfg.n_layers * (cfg.n_heads + 1)
        elif node_name[0] == 'm':
            layer = int(node_name[1:])
            return 1 + layer * (cfg.n_heads + 1) + cfg.n_heads
        elif node_name[0] == 'a':
            layer, head = node_name.split('.')
            layer = int(layer[1:])
            head = int(head[1:])
            i =  1 + layer * (cfg.n_heads + 1)
            return slice(i, i + cfg.n_heads) if attn_slice else i + head
        else:
            raise ValueError(f"Invalid node: {node_name}")

    def forward_index(self, node:Node, attn_slice=True) -> int:
        return Graph._forward_index(self.cfg, node.name, attn_slice)

    @classmethod
    def _backward_index(cls, cfg, node_name:str, qkv=None, attn_slice=False) -> int:
        """Given a model's config and a node specification, return the backward (destination) index of the node in the graph. The backward index is the index of the node in the backward pass of the model, which is used to index into the graph's tensors.

        Args:
            cfg (_type_): A (HookedTransformer) config object
            node_name (str): Name of the node: 'input', 'logits', 'm0', 'a0.h0', etc.
            qkv (_type_, optional): Whether the destination (for attention heads) is the q/k/v input. Defaults to None.
            attn_slice (bool, optional): _description_. Defaults to False.

        Returns:
            int: the backward index
        """
        if node_name == 'input':
            raise ValueError(f"No backward for input node")
        elif node_name == 'logits':
            return -1
        elif node_name[0] == 'm':
            layer = int(node_name[1:])
            return (layer) * (3 * cfg['n_heads'] + 1) + 3 * cfg['n_heads']
        elif node_name[0] == 'a':
            assert qkv in 'qkv', f'Must give qkv for AttentionNode, but got {qkv}'
            layer, head = node_name.split('.')
            layer = int(layer[1:])
            head = int(head[1:])
            i = layer * (3 * cfg['n_heads'] + 1) + ('qkv'.index(qkv) * cfg['n_heads'])
            return slice(i, i + cfg['n_heads']) if attn_slice else i + head
        else:
            raise ValueError(f"Invalid node: {node_name}")

    def backward_index(self, node:Node, qkv=None, attn_slice=True) -> int:
        return Graph._backward_index(self.cfg, node.name, qkv, attn_slice)

    def get_dst_nodes(self) -> List[str]:
        heads = []
        for layer in range(self.cfg['n_layers']):
            for letter in 'qkv':
                for attention_head in range(self.cfg['n_heads']):
                    heads.append(f'a{layer}.h{attention_head}<{letter}>')
            heads.append(f'm{layer}')
        heads.append('logits')
        return heads

    def weighted_edge_count(self) -> float:
        """Generates a count of the edges, weighted by number of neurons included if applicable

        Returns:
            float: weighted edge count
        """
        if self.neurons_in_graph is not None:
            return (einsum(self.in_graph.float(), self.neurons_in_graph.float(), 'forward backward, forward d_model ->') / self.cfg['d_model']).item()
        else:
            return float(self.count_included_edges())

    def count_included_edges(self) -> int:
        return self.in_graph.sum().item()

    def count_included_nodes(self) -> int:
        return self.nodes_in_graph.sum().item()

    def count_included_neurons(self) -> int:
        return self.neurons_in_graph.sum().item()

    def reset(self, empty=True):
        """Resets the graph, setting everything to zero. If empty is False, sets everything to True instead.
        Args:
            empty (bool, optional): If true, removes everything from graph; otherwise adds everything. Defaults to True.
        """
        if empty:
            self.nodes_in_graph *= False
            self.in_graph *= False
            if self.neurons_in_graph is not None:
                self.neurons_in_graph *= False
            if self.positional_edges_in_graph is not None:
                self.positional_edges_in_graph *= False
            # if self.positional_nodes_in_graph is not None:
            #     self.positional_nodes_in_graph *= False
        else:
            self.nodes_in_graph[:] = True
            self.in_graph[:] = True
            self.in_graph &= self.real_edge_mask
            if self.neurons_in_graph is not None:
                self.neurons_in_graph[:] = True
            if self.positional_edges_in_graph is not None:
                self.positional_edges_in_graph[:] = True
            # if self.positional_nodes_in_graph is not None:
            #     self.positional_nodes_in_graph[:] = True

    def apply_threshold(
        self,
        threshold: float,
        absolute: bool=True,
        reset: bool=True,
        level:Literal['edge','node','neuron']='edge',
        positional: bool=False,
        prune=True,
        **prune_kwargs,
    ):
        """Apply a threshold to the graph, setting the in_graph attribute of edges/nodes/neurons to True if the score is above the threshold.
        If a node or neuron has no score, it's assumed to always be in the graph.

        Args:
            threshold (float): the threshold to apply
            absolute (bool): whether to take the absolute value of the scores before applying the threshold
            reset (bool): resets the graph, setting everything to zero, before applying topn.
                Only if reset=True will corresponding outgoing edges be added after neuron and node topn
            level (str, optional): level at which to apply topn. Defaults to 'edge'.
            prune (bool): whether to prune the graph after applying topn"""

        threshold = float(threshold)
        if reset:
            self.reset()

        if level == 'neuron':
            if positional:
                raise NotImplementedError("positional is currently only implemented for edge-level circuits")
            unscored_neurons = torch.isnan(self.neurons_scores)
            neuron_score_copy = self.neurons_scores.clone()
            if absolute:
                neuron_score_copy = torch.abs(neuron_score_copy)

            # We definitely want unscored neurons to be in the graph
            neuron_score_copy[unscored_neurons] = torch.inf
            included_neurons = (neuron_score_copy >= threshold)
            self.neurons_in_graph[:] = included_neurons

            if reset:
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == 'node':
            if positional:
                raise NotImplementedError("positional is currently only implemented for edge-level circuits")
            unscored_nodes =  torch.isnan(self.nodes_scores)
            
            node_score_copy = self.nodes_scores.clone()
            if absolute:
                node_score_copy = torch.abs(node_score_copy)
                
            node_score_copy[unscored_nodes] = torch.inf
            included_nodes = (node_score_copy >= threshold)
            self.nodes_in_graph[:] = included_nodes
                            
            if reset: 
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        elif level == 'edge':
            if positional:
                assert self.positional_scores is not None, "You haven't computed positional scores yet!"
                edge_scores = self.positional_scores.clone()
            else:
                edge_scores = self.scores.clone()

            if absolute:
                edge_scores = torch.abs(edge_scores)

            # masking out the edges that are not real
            edge_scores[...,~self.real_edge_mask] = -torch.inf

            surpass_threshold = edge_scores >= threshold #(pos) forward backward
            if positional:
                self.positional_edges_in_graph = surpass_threshold.to(self.in_graph.device)
            else:
                self.in_graph[:] = surpass_threshold

            if reset:
                positional_nodes_with_outgoing = surpass_threshold.any(dim=-1) #(pos) forward
                positional_nodes_with_ingoing = einsum(
                    surpass_threshold.any(dim=-2).float(),#(pos) backward
                    self.forward_to_backward.float(),
                    '... backward, forward backward -> ... forward'
                ) > 0
                nodes_with_outgoing = reduce(positional_nodes_with_outgoing, '... forward -> forward', 'max')
                nodes_with_ingoing = reduce(positional_nodes_with_ingoing, '... forward -> forward', 'max')
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing
        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune(**prune_kwargs)

    def apply_topn(
        self, n:int,
        absolute: bool=True,
        level:Literal['edge','node','neuron']='edge',
        reset: bool=True,
        positional: bool=False,
        prune:bool=True,
        **prune_kwargs
    ):
        """Sets the graph to contain only the top-n components. The components are specified by the level parameter, which can be 'edge','node', or 'neuron'. If 'node', the top-n nodes are selected based on their scores, and all outgoing edges to nodes in the graph are true. If 'edge', the top-n edges are selected based on their scores. If 'neuron', the top-n neurons are selected based on their scores, and all outgoing edges to nodes with neurons in the graph are true.

        Args:
            n (int): number of edges/nodes/neurons to take
            absolute (bool): whether to apply topn based on the absolute value of the scores
            reset (bool): resets the graph, setting everything to zero, before applying topn. Only if reset=True will corresponding edges be added after neuron and node topn
            level (str, optional): level at which to apply topn. Defaults to 'edge'.
            prune (bool): whether to prune the graph after applying topn
        """
        if reset:
            self.reset()

        if level == 'neuron':
            if positional:
                raise NotImplementedError("positional is currently only implemented for edge-level circuits")
            scored_neurons =  ~torch.isnan(self.neurons_scores)
            n_scored_neurons = scored_neurons.sum()
            assert  n <= n_scored_neurons, f"Requested n ({n}) is greater than the number of scored neurons ({n_scored_neurons})"
            neuron_score_copy = self.neurons_scores.clone()
            if absolute:
                neuron_score_copy = torch.abs(neuron_score_copy)

            neuron_score_copy[~scored_neurons] = -torch.inf
            sorted_neurons = torch.argsort(neuron_score_copy.view(-1), descending=True)

            # set the topn neurons to be in the graph
            self.neurons_in_graph.view(-1)[sorted_neurons[:n]] = True
            # set those outside the topn not to be in the graph
            self.neurons_in_graph.view(-1)[sorted_neurons[n:]] = False
            # unscored neurons must also be re-added to the graph
            self.neurons_in_graph.view(-1)[~scored_neurons.view(-1)] = True
            # remove any nodes with no neurons in the graph
            
            if reset: 
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.nodes_in_graph += self.neurons_in_graph.any(dim=1)
                self.in_graph += self.nodes_in_graph.view(-1, 1)
                
        elif level == 'node':
            if positional:
                raise NotImplementedError("positional is currently only implemented for edge-level circuits")
            scored_nodes =  ~torch.isnan(self.nodes_scores)
            n_scored_nodes = scored_nodes.sum()
            assert  n <= n_scored_nodes, f"Requested n ({n}) is greater than the number of scored nodes ({n_scored_nodes})"

            node_score_copy = self.nodes_scores.clone()
            if absolute:
                node_score_copy = torch.abs(node_score_copy)
                
            node_score_copy[~scored_nodes] = -torch.inf
            sorted_nodes = torch.argsort(node_score_copy.view(-1), descending=True)

            # set the topn neurons to be in the graph
            self.nodes_in_graph.view(-1)[sorted_nodes[:n]] = True
            # set those outside the topn not to be in the graph
            self.nodes_in_graph.view(-1)[sorted_nodes[n:]] = False
            # unscored nodes must also be re-added to the graph
            self.nodes_in_graph.view(-1)[~scored_nodes.view(-1)] = True
                            
            if reset: 
                # if we've reset the graph (everything is empty), add in the nodes that are on
                # and activate their outgoing edges
                self.in_graph += self.nodes_in_graph.view(-1, 1)

        # get top-n edges
        elif level == 'edge':
            assert n <= self.real_edge_mask.sum(), f"Requested n ({n}) is greater than the number of edges ({self.real_edge_mask.sum()})"

            if positional:
                assert self.positional_scores is not None, "You haven't computed positional scores yet!"
                edge_scores = self.positional_scores.clone()
            else:
                edge_scores = self.scores.clone()

            if absolute:
                edge_scores = torch.abs(edge_scores)

            # masking out the edges that are not real
            edge_scores[...,~self.real_edge_mask] = -torch.inf 

            sorted_edges = torch.argsort(edge_scores.view(-1), descending=True)
            if positional:
                self.positional_edges_in_graph.view(-1)[sorted_edges[:n]] = True
                self.positional_edges_in_graph.view(-1)[sorted_edges[n:]] = False
            else:
                self.in_graph.view(-1)[sorted_edges[:n]] = True
                self.in_graph.view(-1)[sorted_edges[n:]] = False

            if reset:
                if positional:
                    nodes_with_outgoing = self.positional_edges_in_graph.any(dim=-1).any(dim=0) #pos forward backward -> pos forward -> forward
                    nodes_with_ingoing = einsum(
                        self.in_graph.any(dim=-2).float(),#pos backward
                        self.forward_to_backward.float(),
                        'pos backward, forward backward -> forward'
                    ) > 0
                else:
                    nodes_with_outgoing = self.in_graph.any(dim=1)
                    nodes_with_ingoing = einsum(
                        self.in_graph.any(dim=0).float(),
                        self.forward_to_backward.float(),
                        'backward, forward backward -> forward'
                    ) > 0
                nodes_with_ingoing[0] = True
                self.nodes_in_graph += nodes_with_outgoing & nodes_with_ingoing

        else:
            raise ValueError(f"Invalid level: {level}")

        if prune:
            self.prune(**prune_kwargs)

    def apply_greedy(self, n_edges:int, absolute: bool = True, reset:bool = True, prune:bool = True, **prune_kwargs):
        """
        Gets the topn edges of the graph using a greedy algorithm that works from the logits up. Only defined over edges
        
        Args:
            n_edges (int): the number of edges to include
            reset (bool): whether to reset the graph before applying the greedy algorithm
            absolute (bool): whether to take the absolute value of the scores before applying the greedy algorithm
        """
        if n_edges > len(self.edges):
            raise ValueError(f"n ({n_edges}) is greater than the number of edges ({len(self.edges)})")
        
        if reset:
            self.nodes_in_graph *= False
            self.in_graph *= False

        def abs_id(s: float):
            return abs(s) if absolute else s

        candidate_edges = sorted([edge for edge in self.edges.values() if edge.child.in_graph], key = lambda edge: abs_id(edge.score), reverse=True)

        edges = heapq.merge(candidate_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        while n_edges > 0:
            n_edges -= 1
            top_edge = next(edges)
            top_edge.in_graph = True
            parent = top_edge.parent
            if not parent.in_graph:
                parent.in_graph = True
                parent_parent_edges = sorted([parent_edge for parent_edge in parent.parent_edges], key = lambda edge: abs_id(edge.score), reverse=True)
                edges = heapq.merge(edges, parent_parent_edges, key = lambda edge: abs_id(edge.score), reverse=True)
        
        if prune:
            self.prune(**prune_kwargs)


    def prune(self, prune_childless:bool=True, prune_parentless:bool=True, positional:bool=False):
        """Converts a potentially messy Graph into one that is fully connected.
        The number of components after this is done is strictly non-increasing; it may remove nodes or edges from the graph, but it won't add them.
        This function first removes nodes with no neurons (if applicable).
        Then, it repeatedly removes nodes that lack incoming or outgoing edges (or both),
        and then edges missing a parent or child.
        Finally, it pruned the neurons of any removed nodes.
        """

        # remove neuronless nodes
        if self.neurons_in_graph is not None:
            self.nodes_in_graph *= self.neurons_in_graph.any(dim=1)

        old_new_same = False
        # Could take twice as many iterations as there are layers! But will probably not
        while not old_new_same:
            # if positional:
            #     nodes_to_keep = self.positional_nodes_in_graph.clone()
            # else:
            nodes_to_keep = self.nodes_in_graph.clone()

            if prune_childless:#remove nodes with 0 outgoing edges
                nodes_with_outgoing = self.in_graph.any(dim=1)
                nodes_to_keep &= nodes_with_outgoing
            if prune_parentless:#remove nodes with 0 incoming edges
                nodes_with_ingoing = einsum(self.in_graph.any(dim=0).float(), self.forward_to_backward.float(), 'backward, forward backward -> forward') > 0
                nodes_with_ingoing[0] = True  # input node always treated as if it has incoming edges
                nodes_to_keep &= nodes_with_ingoing

            # if positional:
            #     old_nodes_in_graph = self.positional_nodes_in_graph.clone()
            #     self.positional_nodes_in_graph = nodes_to_keep
            # else:
            old_nodes_in_graph = self.nodes_in_graph.clone()
            self.nodes_in_graph[:] = nodes_to_keep

            # remove edges with missing parents or children
            # if positional:
            #     forward_in_graph = self.positional_nodes_in_graph.float()
            # else:
            forward_in_graph = self.nodes_in_graph.float()
            backward_in_graph = forward_in_graph @ self.forward_to_backward.float()
            backward_in_graph[...,-1] = 1  # logits node is always present
            edge_remask = einsum(forward_in_graph, backward_in_graph, '... forward, ... backward -> ... forward backward') > 0
            if positional:
                assert self.positional_edges_in_graph is not None
                old_edges_in_graph = self.positional_edges_in_graph.clone()
                self.positional_edges_in_graph *= edge_remask
                # remove edges into / out of MLP nodes if no outgoing / incoming edge belongs to the same position
                # remove edges into attention nodes if all outgoing edges belong to earlier positions, and vice-versa
                edges_into_mlp = einsum(
                    self.positional_edges_in_graph, self.mlp_mask_backward,
                    "pos forward backward, backward -> pos forward backward"
                )
                edges_from_mlp = einsum(
                    self.positional_edges_in_graph, self.mlp_mask_forward,
                    "pos forward backward, forward -> pos forward backward"
                )
                #TODO possibly make attention masks into graph attributes
                edges_into_attn = einsum(
                    self.positional_edges_in_graph, ~self.mlp_mask_backward,
                    "pos forward backward, backward -> pos forward backward"
                )
                edges_into_attn[...,-1] = 0
                edges_from_attn = einsum(
                    self.positional_edges_in_graph, ~self.mlp_mask_forward,
                    "pos forward backward, forward -> pos forward backward"
                )
                edges_from_attn[:,0,:] = 0
                if prune_childless:
                    #remove edges INTO nodes if outgoing edges don't fit
                    #mlp nodes
                    edges_from_mlp_with_backward_indexing = einsum(
                        edges_from_mlp, self.forward_to_backward,
                        "pos forward_mlp backward, forward_mlp backward_mlp -> pos backward_mlp"
                    ) > 0
                    keep_edges_into_mlp = einsum(
                        edges_into_mlp, edges_from_mlp_with_backward_indexing,
                        "pos forward backward_mlp, pos backward_mlp -> pos forward backward_mlp"
                    ) > 0
                    #attention nodes
                    edges_from_attn_with_backward_indexing = einsum(
                        edges_from_attn, self.forward_to_backward,
                        "pos forward_attn backward, forward_attn backward_attn -> pos backward_attn"
                    ) > 0
                    #edges from attn at positions higher or equal to current one (the only valid ones);
                    # doing reverse cumsum as in https://github.com/pytorch/pytorch/issues/33520
                    cum_sum = torch.cumsum(edges_from_attn_with_backward_indexing, dim=0)
                    re_cum_sum = (edges_from_attn_with_backward_indexing - cum_sum + cum_sum[-1:None]) > 0
                    keep_edges_into_attn = einsum(
                        edges_into_attn, re_cum_sum,
                        "pos forward backward_attn, pos backward_attn backward -> pos forward backward_attn"
                    ) > 0
                    self.positional_edges_in_graph &= (keep_edges_into_mlp | keep_edges_into_attn)
                if prune_parentless:
                    #remove edges OUT OF nodes if incoming edges don't fit
                    #mlp nodes
                    edges_into_mlp_with_forward_indexing = einsum(
                        edges_into_mlp, self.forward_to_backward,
                        "pos forward backward_mlp, forward_mlp backward_mlp -> pos forward_mlp"
                    ) > 0
                    keep_edges_from_mlp = einsum(
                        edges_from_mlp, edges_into_mlp_with_forward_indexing,
                        "pos forward_mlp backward, pos forward_mlp -> pos forward_mlp backward"
                    ) > 0
                    #attention nodes
                    edges_into_attn_with_forward_indexing = einsum(
                        edges_into_attn, self.forward_to_backward,
                        "pos forward backward_attn, forward_attn backward_attn -> pos forward_attn"
                    ) > 0
                    #edges into attn at positions lower or equal to current one (the only valid ones);
                    cum_sum = torch.cumsum(edges_into_attn_with_forward_indexing, dim=0) > 0
                    keep_edges_from_attn = einsum(
                        edges_from_attn, cum_sum,
                        "pos forward_attn backward, pos forward_attn -> pos forward_attn backward"
                    ) > 0
                    self.positional_edges_in_graph &= (keep_edges_from_mlp | keep_edges_from_attn)

                old_new_same = (
                    torch.all(old_nodes_in_graph == self.nodes_in_graph) and
                    torch.all(old_edges_in_graph == self.positional_edges_in_graph)
                )
            else:
                old_edges_in_graph = self.in_graph.clone()
                self.in_graph *= edge_remask
                old_new_same = (
                    torch.all(old_nodes_in_graph == self.nodes_in_graph) and
                    torch.all(old_edges_in_graph == self.in_graph)
                )

        # remove neurons from nodes not in the graph
        if self.neurons_in_graph is not None:
                self.neurons_in_graph *= self.nodes_in_graph.view(-1, 1)


    @classmethod
    def from_model(
        cls, model_or_config: Union[HookedTransformer,HookedTransformerConfig, Dict],
        neuron_level: bool = False, node_scores: bool = False,
        #n_pos:int=0,
    ) -> 'Graph':
        """Instantiate a Graph object from a HookedTransformer or HookedTransformerConfig object, or a similar Dict.
        The neuron_level parameter determines whether the graph should be neuron-level or not,
        while the node_scores parameter determines whether the graph should have node scores or not.
        If you don't have scores for all nodes / neurons, just don't set them (default is torch.nan).
        Any node/neuron without a real score will always be kept in the graph when doing node/neuron-level topn
        (but might be eliminated by another level's topn, e.g. a node with no neuron scores might be removed if it loses all edges)

        Args:
            model_or_config (Union[HookedTransformer,HookedTransformerConfig, Dict]):
                A config object; it needs to contain n_layers, n_heads, parallel_attn_mlp, and d_model.
            neuron_level (bool, optional): _description_. Defaults to False.
            node_scores (bool, optional): _description_. Defaults to False.

        Raises:
            ValueError: If you pass an invalid type for model_or_config

        Returns:
            _type_: a Graph
        """
        graph = Graph()
        graph.cfg = GraphConfig()
        if isinstance(model_or_config, HookedTransformer):
            cfg = model_or_config.cfg
            graph.cfg.update({
                'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'd_model': cfg.d_model
            })
        elif isinstance(model_or_config, HookedTransformerConfig):
            cfg = model_or_config
            graph.cfg.update({
                'n_layers': cfg.n_layers, 'n_heads': cfg.n_heads, 'parallel_attn_mlp':cfg.parallel_attn_mlp, 'd_model': cfg.d_model
            })
        elif isinstance(model_or_config, dict):
            graph.cfg.update(model_or_config)
        else:
            raise ValueError(f"Invalid input type: {type(model_or_config)}")

        graph.n_forward = 1 + graph.cfg['n_layers'] * (graph.cfg['n_heads'] + 1)
        graph.n_backward = graph.cfg['n_layers'] * (3 * graph.cfg['n_heads'] + 1) + 1
        graph.forward_to_backward = torch.zeros((graph.n_forward, graph.n_backward)).bool()

        graph.scores = torch.zeros((graph.n_forward, graph.n_backward))# if n_pos==0
            #else torch.zeros((n_pos, graph.n_forward, graph.n_backward))
        #)
        graph.real_edge_mask = torch.zeros_like(graph.scores).bool()
        graph.in_graph = torch.zeros_like(graph.scores).bool()
        graph.nodes_in_graph = torch.zeros(graph.n_forward).bool()# if n_pos==0
            #else torch.zeros((n_pos, graph.n_forward))
        #)
        graph.mlp_mask_forward = torch.zeros(graph.n_forward).bool()
        graph.mlp_mask_backward = torch.zeros(graph.n_backward).bool()
        graph.positional_scores = None
        graph.positional_edges_in_graph = None
        #graph.positional_nodes_in_graph = None
        if node_scores:
            graph.nodes_scores = torch.zeros_like(graph.nodes_in_graph).float()
            graph.nodes_scores[:] = torch.nan
        else:
            graph.nodes_scores = None
        if neuron_level:
            graph.neurons_scores = torch.zeros((graph.n_forward, graph.cfg['d_model']))# if n_pos==0
                #else torch.zeros((n_pos, graph.n_forward, graph.cfg['d_model']))
            #)
            graph.neurons_in_graph = torch.zeros_like(graph.neurons_scores).bool()
            graph.neurons_scores[:] = torch.nan
        else:
            graph.neurons_in_graph = None
            graph.neurons_scores = None
        # if n_pos>0:
        #     graph.positional_scores = torch.zeros((n_pos, graph.n_forward, graph.n_backward))
        # else:
        #     graph.positional_scores = None

        input_node = InputNode(graph)
        graph.nodes[input_node.name] = input_node
        residual_stream = [input_node]

        for layer in range(graph.cfg['n_layers']):
            attn_nodes = [AttentionNode(layer, head, graph) for head in range(graph.cfg['n_heads'])]
            mlp_node = MLPNode(layer, graph)

            for attn_node in attn_nodes:
                graph.nodes[attn_node.name] = attn_node
                for letter in 'qkv':
                    graph.forward_to_backward[
                        graph.forward_index(attn_node, attn_slice=False), graph.backward_index(attn_node, attn_slice=False, qkv=letter)
                    ] = True
            graph.nodes[mlp_node.name] = mlp_node
            graph.forward_to_backward[
                graph.forward_index(mlp_node, attn_slice=False), graph.backward_index(mlp_node, attn_slice=False)
            ] = True
            graph.mlp_mask_forward[graph.forward_index(mlp_node, attn_slice=False)] = True
            graph.mlp_mask_backward[graph.backward_index(mlp_node, attn_slice=False)] = True

            if graph.cfg['parallel_attn_mlp']:
                for node in residual_stream:
                    for attn_node in attn_nodes:          
                        for letter in 'qkv':
                            graph.add_edge(node, attn_node, qkv=letter)
                    graph.add_edge(node, mlp_node)

                residual_stream += attn_nodes
                residual_stream.append(mlp_node)

            else:
                for node in residual_stream:
                    for attn_node in attn_nodes:
                        for letter in 'qkv':
                            graph.add_edge(node, attn_node, qkv=letter)
                residual_stream += attn_nodes

                for node in residual_stream:
                    graph.add_edge(node, mlp_node)
                residual_stream.append(mlp_node)

        logit_node = LogitNode(graph.cfg['n_layers'], graph)#we assume the metric is just based on the final logits
        for node in residual_stream:
            graph.add_edge(node, logit_node)

        graph.nodes[logit_node.name] = logit_node

        return graph


    def to_json(self, filename: str):
        if self.positional_scores is not None:
            raise NotImplementedError("Positional scores to json is not supported (and probably will not be as it is memory-inefficient).")
        # non serializable info
        d = {'cfg':dict(self.cfg)}
        node_dict = {}
        for node_name, node in self.nodes.items():
            node_dict[node_name] = {'in_graph': bool(node.in_graph)}
            if self.nodes_scores is not None:
                node_dict[node_name]['score'] = float(node.score)
            if self.neurons_in_graph is not None:
                node_dict[node_name]['neurons'] = self.neurons_in_graph[self.forward_index(node)].tolist()
                node_dict[node_name]['neurons_scores'] = self.neurons_scores[self.forward_index(node)].tolist()
        d['nodes'] = node_dict

        edge_dict = {}
        for edge_name, edge in self.edges.items():
            edge_dict[edge_name] = {'score': edge.score.item(), 'in_graph': bool(edge.in_graph)}

        d['edges'] = edge_dict

        with open(filename, 'w') as f:
            json.dump(d, f)


    def to_pt(self, filename: str):
        """Export this Graph as a .pt file

        Args:
            filename (str): The filename to save the graph to
        """
        src_nodes = [node.name for node in self.nodes.values() if not isinstance(node, LogitNode)]
        dst_nodes = self.get_dst_nodes()
        d = {'cfg':dict(self.cfg), 'src_nodes': src_nodes, 'dst_nodes': dst_nodes, 'edges_scores': self.scores, 'edges_in_graph': self.in_graph, 'nodes_in_graph': self.nodes_in_graph}
        if self.nodes_scores is not None:
            d['nodes_scores'] = self.nodes_scores
        if self.neurons_in_graph is not None:
            d['neurons_in_graph'] = self.neurons_in_graph
            d['neurons_scores'] = self.neurons_scores
        if self.positional_scores is not None:
            d['positional_scores'] = self.positional_scores
        if self.positional_edges_in_graph is not None:
            d['positional_edges_in_graphs'] = self.positional_edges_in_graph
        torch.save(d, filename)

    @classmethod
    def from_json(cls, json_path: str) -> 'Graph':
        """
        Load a Graph object from a JSON file.
        The JSON should have the following keys:
            1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
            2. 'nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
            3. 'edges': Dict[str, Dict] which maps an edge name ('node->node') to a dictionary contains values 
            4. 'neurons': Optional[Dict[str, List[bool]]] which maps a node name (i.e. 'm11' or 'a0.h11') to a list of boolean values, indicating which of its neurons are part of the circuit.

        NOTE: This method isn't disk-space efficient, and shouldn't be used when the circuits contains edges between neuron-resolution nodes.
        """
        with open(json_path, 'r') as f:
            d = json.load(f)
            assert all([k in d.keys() for k in ['cfg', 'nodes', 'edges']]), "Bad input JSON format - Missing keys"

        g = Graph.from_model(d['cfg'], neuron_level=True, node_scores=True)
        any_node_scores, any_neurons, any_neurons_scores = False, False, False
        for name, node_dict in d['nodes'].items():
            if name == 'logits':
                continue
            g.nodes[name].in_graph = node_dict['in_graph']
            if 'score' in node_dict:
                any_node_scores = True
                g.nodes[name].score = node_dict['score']
            if 'neurons' in node_dict:
                any_neurons = True
                g.neurons_in_graph[g.forward_index(g.nodes[name])] = torch.tensor(node_dict['neurons']).float()
            if 'neurons_scores' in node_dict:
                any_neurons_scores = True
                g.neurons_scores[g.forward_index(g.nodes[name])] = torch.tensor(node_dict['neurons_scores']).float()

        if not any_node_scores:
            g.nodes_scores = None
        if not any_neurons:
            g.neurons_in_graph = None
        if not any_neurons_scores:
            g.neurons_scores = None

        for name, info in d['edges'].items():
            g.edges[name].score = info['score']
            g.edges[name].in_graph = info['in_graph']

        return g

    @classmethod
    def from_pt(cls, pt_path: str) -> 'Graph':
        """
        Load a graph object from a pytorch-serialized file.
        The file should contain a dict with the following items -
            1. 'cfg': Configuration dictionary, containing similar values to a TLens configuration object.
            2. 'src_nodes': Dict[str, bool] which maps a node name (i.e. 'm11' or 'a0.h11') to a boolean value, indicating if the node is part of the circuit.
            3. 'dst_nodes': List[str] containing the names of the possible destination nodes, in the same order as the edges tensor.
            4. 'edges': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents the edge score between the src node and dst node.
            5. 'edges_in_graph': torch.tensor[n_src_nodes, n_dst_nodes], where each value in (src, dst) represents if the edge is in the graph or not.
            6. 'neurons': [Optional] torch.tensor[n_src_nodes, d_model], where each value in (src, neuron) indicates whether the neuron is in the graph or not
        """
        d = torch.load(pt_path)
        required_keys = ['cfg', 'src_nodes', 'dst_nodes', 'edges_scores', 'edges_in_graph', 'nodes_in_graph']
        assert all([k in d.keys() for k in required_keys]), f"Bad torch circuit file format. Found keys - {d.keys()}, missing keys - {set(required_keys) - set(d.keys())}"
        assert d['edges_scores'].shape == d['edges_in_graph'].shape, "Bad edges array shape"

        g = Graph.from_model(d['cfg'])

        g.in_graph[:] = d['edges_in_graph']
        g.scores[:] = d['edges_scores']
        g.nodes_in_graph[:] = d['nodes_in_graph']

        if 'nodes_scores' in d:
            g.nodes_scores = d['nodes_scores']

        if 'neurons_in_graph' in d:
            g.neurons_in_graph = d['neurons_in_graph']

        if 'neurons_scores' in d:
            g.neurons_scores = d['neurons_scores']

        if 'positional_scores' in d:
            g.positional_scores = d['positional_scores']

        if 'positional_edges_in_graph' in d:
            g.positional_edges_in_graph = d['positional_edges_in_graph']

        return g

    def to_image(
        self,
        filename:str,
        colorscheme: str = "Pastel2",
        minimum_penwidth: float = 0.6,
        maximum_penwidth: float = 5.0,
        layout: str="dot",
        seed: Optional[int] = None
    ):

        """Export the graph as a .png file

        Filename: the filename to save the graph to
        Colorscheme: a cmap colorscheme
        """
        if self.positional_scores is not None:
            raise NotImplementedError(
                "Cannot yet represent a positional circuit as image. We're working on it!"
            )#TODO to_image for positional_scores
        import pygraphviz as pgv
        g = pgv.AGraph(directed=True, bgcolor="white", overlap="false", splines="true", layout=layout)

        if seed is not None:
            np.random.seed(seed)

        colors = {node.name: generate_random_color(colorscheme) for node in self.nodes.values()}

        for node in self.nodes.values():
            if node.in_graph:
                g.add_node(node.name,
                        fillcolor=colors[node.name],
                        color="black",
                        style="filled, rounded",
                        shape="box",
                        fontname="Helvetica",
                        )

        scores = self.scores.view(-1).abs()
        max_score = scores.max().item()
        min_score = scores.min().item()
        for edge in self.edges.values():
            if edge.in_graph:
                normalized_score = (abs(edge.score) - min_score) / (max_score - min_score) if max_score != min_score else abs(edge.score)
                penwidth = max(minimum_penwidth, normalized_score * maximum_penwidth)
                g.add_edge(edge.parent.name,
                        edge.child.name,
                        penwidth=str(penwidth),
                        color=get_color(edge.qkv, edge.score),
                        )
        g.draw(filename, prog="dot")
