import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple, NamedTuple
from torch_geometric.utils import add_self_loops
from torch_geometric.transforms import GDC
from torch_sparse import coalesce

def compute_ppr(edge_index, edge_weight=None, alpha=0.2, eps=0.1, ignore_edge_attr=True, add_self_loop=True):
    N = edge_index.max().item() + 1
    if ignore_edge_attr or edge_weight is None:
        edge_weight = torch.ones(
            edge_index.size(1), device=edge_index.device)
    if add_self_loop:
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1, num_nodes=N)
        edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')
    diff_mat = GDC().diffusion_matrix_exact(
        edge_index, edge_weight, N, method='ppr', alpha=alpha)
    edge_index, edge_weight = GDC().sparsify_dense(diff_mat, method='threshold', eps=eps)
    edge_index, edge_weight = coalesce(edge_index, edge_weight, N, N)
    edge_index, edge_weight = GDC().transition_matrix(
        edge_index, edge_weight, N, normalization='sym')

    return edge_index, edge_weight


class Graph(NamedTuple):
    x: torch.FloatTensor
    edge_index: torch.LongTensor
    edge_weights: Optional[torch.FloatTensor]

    def unfold(self) -> Tuple[torch.FloatTensor, torch.LongTensor, Optional[torch.FloatTensor]]:
        return self.x, self.edge_index, self.edge_weights


class Augmentor(ABC):
    """Base class for graph augmentors."""
    def __init__(self):
        pass

    @abstractmethod
    def augment(self, g: Graph) -> Graph:
        raise NotImplementedError(f"GraphAug.augment should be implemented.")

    def __call__(
            self, x: torch.FloatTensor,
            edge_index: torch.LongTensor, edge_weight: Optional[torch.FloatTensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.augment(Graph(x, edge_index, edge_weight)).unfold()



class PPRDiffusion(Augmentor):
    def __init__(self, alpha: float = 0.2, eps: float = 1e-4, use_cache: bool = True, add_self_loop: bool = True):
        super(PPRDiffusion, self).__init__()
        self.alpha = alpha
        self.eps = eps
        self._cache = None
        self.use_cache = use_cache
        self.add_self_loop = add_self_loop

    def augment(self, g: Graph) -> Graph:
        if self._cache is not None and self.use_cache:
            return self._cache
        x, edge_index, edge_weights = g.unfold()
        edge_index, edge_weights = compute_ppr(
            edge_index, edge_weights,
            alpha=self.alpha, eps=self.eps, ignore_edge_attr=False, add_self_loop=self.add_self_loop
        )
        res = Graph(x=x, edge_index=edge_index, edge_weights=edge_weights)
        self._cache = res
        return res