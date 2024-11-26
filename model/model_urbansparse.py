import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.utils.cpp_extension import load
from tqdm import tqdm
from sampler import get_sampler, drop_node


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

class RegionInfoContrast(torch.nn.Module):
    def __init__(self, loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(RegionInfoContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, h1_batch=None, h2_batch=None,
                extra_pos_mask=None, extra_neg_mask=None):
        anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=h2_batch)
        anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=h1_batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5

class MLP(nn.Module):
    def __init__(self, in_ft, out_ft):
        super(MLP, self).__init__()
        self.ffn = nn.Sequential(
            nn.Linear(in_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU(),
            nn.Linear(out_ft, out_ft),
            nn.PReLU()
        )
        self.linear_shortcut = nn.Linear(in_ft, out_ft)

    def forward(self, x):
        return self.ffn(x) + self.linear_shortcut(x)
    
class GConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GConv, self).__init__()
        self.layers = nn.ModuleList()
        self.activation = nn.PReLU(hidden_dim)
        for i in range(num_layers):
            if i == 0:
                self.layers.append(GCNConv(input_dim, hidden_dim))
            else:
                self.layers.append(GCNConv(hidden_dim, hidden_dim))

    def forward(self, x, edge_index, batch):
        z = x
        zs = []
        for conv in self.layers:
            z = conv(z, edge_index)
            z = self.activation(z)
            zs.append(z)
        gs = [global_mean_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g


class EvaluationLayer(nn.Module):
    '''
        The nn.Embedding is too slow.
        We use this to replace it.
    '''
    def __init__(self, expanded_dim, hidden_dim):
        super(EvaluationLayer, self).__init__()
        self.expanded_dim = expanded_dim
        self.hidden_dim = hidden_dim
        ablation_random_init = False
        # The effect of zero initialization is marginal.
        # However, it is necessary for the model to be deployed without labeled queries.
        if ablation_random_init:
            self.weight = nn.Parameter(torch.randn(expanded_dim, hidden_dim, dtype=torch.float32))
        else:
            self.weight = nn.Parameter(torch.zeros(expanded_dim, hidden_dim, dtype=torch.float32))

    def forward(self, x):
        # assert torch.all(self.weight.data[-1,:] == 0)
        out = self.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))



class UrbanSparse(nn.Module):
    def __init__(self, bloom_filter_dim=8192, hidden_1=512, hidden_2=64, dense_dim=64, no_ret=False):
        super(UrbanSparse, self).__init__()
        self.bloom_filter_dim = bloom_filter_dim
        self.expanded_dim = bloom_filter_dim + 1
        self.embedding_dim = hidden_1
        self.codebook= nn.EmbeddingBag(self.expanded_dim, hidden_1, mode='mean', padding_idx=bloom_filter_dim)

        # Only for retrieval tasks.
        self.query_hidden_layers = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.PReLU(),
            nn.Linear(hidden_2, hidden_2),
            nn.PReLU(),
        )
        self.query_evaluator = EvaluationLayer(self.expanded_dim, hidden_2)
        self.poi_evaluator = EvaluationLayer(self.expanded_dim, hidden_2)

        self.poi_hidden_layers = nn.Sequential(
            nn.PReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.PReLU(),
            nn.Linear(hidden_2, hidden_2),
            nn.PReLU(),
        )

        self.collective_activation = nn.PReLU()
        self.individual_activation = nn.PReLU()

        # Distance Modeling
        self.a = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'))
        self.b = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))
        self.c = nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device='cuda'))
        self.d = nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device='cuda'))

        if not no_ret:
            self.isin_cuda = load(
                name="isin_cuda",
                sources=["cuda/isin_cuda.cu"],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-lineinfo'],
                verbose=True
            )

            self.text_sim_cuda = load(
                name="text_sim_cuda",
                sources=["cuda/text_sim_cuda.cu"],
                extra_cflags=['-O3'],
                extra_cuda_cflags=['-lineinfo'],
                verbose=True
            )

        self.mlp1 = MLP(dense_dim, dense_dim)
        self.mlp2 = MLP(2 * dense_dim, dense_dim)
        self.gnn1 = GConv(hidden_1, dense_dim, 2)
        self.gnn2 = GConv(hidden_1, dense_dim, 2)

    @torch.no_grad()
    def encode_query(self, query_bloom_filter: torch.LongTensor):
        query_embedding = self.codebook(query_bloom_filter)
        query_hidden = self.query_hidden_layers(query_embedding)
        selected_columns = self.query_evaluator(query_bloom_filter)
        query_sparse_weights = self.individual_activation(query_hidden.unsqueeze(1).mul(selected_columns).sum(dim=-1)) + 1
        return query_sparse_weights
    
    @torch.no_grad()
    def encode_poi(self, poi_bloom_filter: torch.LongTensor):
        poi_embedding = self.codebook(poi_bloom_filter)
        poi_hidden = self.poi_hidden_layers(poi_embedding)
        selected_columns = self.poi_evaluator(poi_bloom_filter)
        poi_sparse_weights = self.individual_activation(poi_hidden.unsqueeze(1).mul(selected_columns).sum(dim=-1)) + 1
        return poi_sparse_weights
    
    @torch.no_grad()
    def brute_force_search(self, query_bloom_filter: torch.LongTensor, query_weights: torch.Tensor, query_loc: torch.Tensor, poi_bloom_filter: torch.LongTensor, poi_weights: torch.Tensor, poi_loc: torch.Tensor):
        '''
        at::Tensor text_sim(
            at::Tensor elements,
            at::Tensor test_elements,
            at::Tensor element_weights,
            at::Tensor test_element_weights,
            int padding_idx);
        '''
        text_sim =  self.text_sim_cuda.text_sim(query_bloom_filter, poi_bloom_filter, query_weights, poi_weights, self.bloom_filter_dim)
        # select top_k indices from text_sim
        # compute relevance score
        dist = torch.sum((query_loc.unsqueeze(1) - poi_loc) ** 2, dim=-1).sqrt() # [B, K]
        dist_sim = - torch.log(dist + 1)
        max_sim = text_sim.max(dim=-1, keepdim=True)[0]
        text_sim = (2* text_sim - max_sim) / (max_sim + 1e-6)
        relevance_score = (self.c - (self.a * text_sim + self.b).sigmoid()) * (dist_sim - self.d)
        return relevance_score


    def bi_encode(self, query_bloom_filter: torch.LongTensor, query_loc: torch.Tensor, poi_bloom_filter: torch.LongTensor, poi_loc: torch.Tensor):
        '''
        Fast training via pre-intersection and custom cuda kernel.
        '''
        B, K, _ = poi_bloom_filter.shape
        query_embedding = self.codebook(query_bloom_filter)
        query_hidden = self.query_hidden_layers(query_embedding) # B, H
        poi_embedding = self.codebook(poi_bloom_filter.view(-1, poi_bloom_filter.shape[-1])).view(B, K, -1)
        poi_hidden = self.poi_hidden_layers(poi_embedding) # B, K, H

        # Pre-intersection to save training time.
        intersection = query_bloom_filter.unsqueeze(1).expand(-1, K, -1)
        intersection_mask = self.isin_cuda.isin_cuda(intersection.reshape(B * K, -1), poi_bloom_filter.view(B * K, -1), self.bloom_filter_dim, False).view(B, K, -1)
        intersection = intersection.masked_fill(~intersection_mask, self.bloom_filter_dim).view(B, K, -1)

        query_selected_columns = self.query_evaluator(intersection) # B, K, L, H
        poi_selected_columns = self.poi_evaluator(intersection) # B, K, L, H
        
        # Separate the sparse embeddings so that we can save them for inference speed.
        query_sparse_weights = self.individual_activation(query_hidden.unsqueeze(1).unsqueeze(2).mul(query_selected_columns).sum(dim=-1)) + 1 # B, K, L
        poi_sparse_weights = self.individual_activation(poi_hidden.unsqueeze(2).mul(poi_selected_columns).sum(dim=-1)) + 1 # B, K, L

        text_sim = query_sparse_weights * poi_sparse_weights # B, K, L
        text_sim = text_sim.masked_fill(~intersection_mask, 0)
        text_sim = text_sim.sum(dim=-1) # B, K
        
        dist = torch.sum((query_loc.unsqueeze(1) - poi_loc) ** 2, dim=-1).sqrt() # [B, K]
        dist_sim = - torch.log(dist + 1)

        # ablation study: no text normalization
        # without text normalization the model crashed and cannot be trained.
        ablation_no_text_norm = False
        if not ablation_no_text_norm:
            max_sim = text_sim.max(dim=-1, keepdim=True)[0]
            text_sim = (2* text_sim - max_sim) / (max_sim + 1e-6)

        relevance_score = (self.c - (self.a * text_sim + self.b).sigmoid()) * (dist_sim - self.d)
        return relevance_score
    
    def region_contrast(self, local_x: torch.LongTensor, global_x: torch.LongTensor, local_edge_index: torch.LongTensor, global_edge_index: torch.LongTensor, local_batch: torch.LongTensor, global_batch: torch.LongTensor):
        
        local_x_emb = self.codebook(local_x)
        global_x_emb = self.codebook(global_x)
        # activation to increase the non-linearity
        local_x_emb = self.collective_activation(local_x_emb)
        global_x_emb = self.collective_activation(global_x_emb)
        # if the model is not in eval mode, we drop some nodes
        # We empirically found that this works similar to drop the bits in the bloom filter.
        if self.training:
            global_edge_index, _ = drop_node(global_edge_index, None, keep_prob=0.8)

        local_x_emb, local_g = self.gnn1(local_x_emb, local_edge_index, local_batch)
        global_x_emb, global_g = self.gnn2(global_x_emb, global_edge_index, global_batch)

        local_x_emb = self.mlp1(local_x_emb)
        global_x_emb = self.mlp1(global_x_emb)

        local_g = self.mlp2(local_g)
        global_g = self.mlp2(global_g)

        return local_x_emb, local_g, global_x_emb, global_g
    

if __name__ == "__main__":
    # We now test the brute force search.

    import argparse
    import time
    from torch.utils.data import DataLoader
    from utils import eval_search
    from dataset import POIDataset

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    args = parser.parse_args()
    dataset = args.dataset
    num_slices = 2
    num_bits = 4096
    padding_idx = num_slices * num_bits
    poi_dataset = POIDataset(dataset, num_slices=num_slices, num_bits=num_bits, load_query=True)
    model = UrbanSparse(bloom_filter_dim=num_slices * num_bits, hidden_1=256, hidden_2=32, dense_dim=64)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters in UrbanSparse: {total_params}")
    
    query_dataloader = DataLoader(poi_dataset.train_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=poi_dataset.train_dataset.collate_fn)
    raw_poi_bloom_filters, raw_poi_locs = poi_dataset.poi_bloom_filters, poi_dataset.poi_locs


    # Brute-force search
    max_len = max([len(x) for x in raw_poi_bloom_filters])
    poi_bloom_filter = torch.empty(len(raw_poi_bloom_filters), max_len, dtype=torch.int32)
    for i, bf in enumerate(raw_poi_bloom_filters):
        poi_bloom_filter[i, :len(bf)] = torch.tensor(bf, dtype=torch.int32)
        poi_bloom_filter[i, len(bf):] = padding_idx
    poi_bloom_filter = poi_bloom_filter.cuda()
    poi_locs = torch.tensor(raw_poi_locs, dtype=torch.float32).cuda()
    poi_weights = torch.ones(len(raw_poi_bloom_filters), max_len, dtype=torch.float32).cuda()

    relevance_scores = []
    query_truths = []
    
    start_time = time.time()
    for data in tqdm(query_dataloader, desc="Brute force search"):
        query_bloom_filter, query_loc, query_truth, query_weights = data
        query_bloom_filter = query_bloom_filter.cuda().to(torch.int32)
        query_bloom_filter[query_bloom_filter == -1] = padding_idx
        query_loc = query_loc.cuda()
        query_weights = query_weights.cuda()
        relevance_score = model.brute_force_search(query_bloom_filter, query_weights, query_loc, poi_bloom_filter, poi_weights, poi_locs)
        top_k_indices = torch.topk(relevance_score, 20, dim=-1, largest=True)[1]
        relevance_scores.append(top_k_indices.cpu())
        query_truths.extend(query_truth)
    relevance_scores = torch.cat(relevance_scores, dim=0).numpy()
    end_time = time.time()
    print(f"Brute force search time: {end_time - start_time} seconds")
    print(f"Number of queries: {len(query_truths)}")
    print(f"Number of POIs: {len(poi_bloom_filter)}")
    print(f"Query Per Second: {len(query_truths) / (end_time - start_time)}")
    eval_results = eval_search(relevance_scores, query_truths)
    print(eval_results)
    

    






