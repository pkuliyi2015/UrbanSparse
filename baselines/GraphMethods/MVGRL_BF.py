# This script tests unsupervised Deep Graph Infomax, with the Bloom filters as the input features.
import argparse
import torch
import numpy as np
import torch.utils.data
import torch_geometric

from torch import nn
from tqdm import tqdm
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import Delaunay
from torch_geometric.nn import global_add_pool

from augmentor import PPRDiffusion
from model.loss import JSD
from model.dataset import POIDataset
from model.utils import load_region_data, evaluate_rf_cv, extract_ids_and_labels
from model.sampler import get_sampler
from statistics import mean, stdev

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='Beijing', help='city name')
argparser.add_argument('--batch_size', type=int, default=16, help='batch size')
args = argparser.parse_args()
dataset = args.dataset


poi_dataset = POIDataset(dataset, num_slices=2, num_bits=4096, load_query=False)
poi_locations = poi_dataset.poi_locs

raw_poi_bloom_filters, raw_poi_locs = poi_dataset.poi_bloom_filters, poi_dataset.poi_locs

    # Brute-force search
max_len = max([len(x) for x in raw_poi_bloom_filters])
poi_bloom_filter = torch.empty(len(raw_poi_bloom_filters), max_len, dtype=torch.int32)
for i, bf in enumerate(raw_poi_bloom_filters):
    poi_bloom_filter[i, :len(bf)] = torch.tensor(bf, dtype=torch.int32)
    poi_bloom_filter[i, len(bf):] = 8192
poi_bloom_filter = poi_bloom_filter.cuda()
poi_features = poi_bloom_filter


region_data = load_region_data(dataset)

# The region data is a dict of dicts.
# The outer dict is keyed by the region id.
# The inner dict is keyed by the 'pois', 'population', 'house_price'
# The value of 'pois' is a list of dicts. Each dict contains the 'index' and 'location' of a POI.
# We now count the total number of POIs in the region data.
total_pois = 0
for region_id, region_info in region_data.items():
    total_pois += len(region_info['pois'])

print(f'Total POIs in Regions: {total_pois}')

# Create a single large graph for all POIs
poi_locations = np.array(poi_locations)
if poi_locations.shape[1] != 2:
    raise ValueError("POI locations must be 2D for Delaunay triangulation")

tri = Delaunay(poi_locations)
edges = set()
for simplex in tri.simplices:
    edges.update((simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3))
    edges.update((simplex[j], simplex[i]) for i in range(3) for j in range(i + 1, 3))
edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

# Create the large graph
# The x is an index vector of the poi_features as the bloom filter is too large to fit in memory
large_graph = Data(x=torch.arange(poi_features.shape[0], dtype=torch.long), edge_index=edge_index)

def create_subgraph_from_region(region_info, large_graph):
    if not region_info['pois']:
        raise ValueError("No POIs found in the region")
    
    subset = [poi['index'] for poi in region_info['pois']]
    
    sub_edge_index = torch_geometric.utils.subgraph(subset, large_graph.edge_index, relabel_nodes=True)[0]
    # Select the features of the nodes in the subset from large_graph.x
    sub_x = large_graph.x[subset]
    
    # Return the subgraph data
    return Data(x=sub_x, edge_index=sub_edge_index)

# Create subgraphs for each region
graphs = []
regions = []
for region_id, region_info in tqdm(region_data.items(), desc='Creating Subgraphs'):
    if region_info['pois']:
        graphs.append(create_subgraph_from_region(region_info, large_graph))
        regions.append(region_info)
print(f'Created {len(graphs)} subgraphs')

'''
Standard MVGRL from PyGCL without modifications.
'''

# Initialize the DGI model. Default settings 
in_channels = poi_features.shape[1]
hidden_channels = 512
out_channels = 512
num_layers = 2


def add_extra_mask(pos_mask, neg_mask=None, extra_pos_mask=None, extra_neg_mask=None):
    if extra_pos_mask is not None:
        pos_mask = torch.bitwise_or(pos_mask.bool(), extra_pos_mask.bool()).float()
    if extra_neg_mask is not None:
        neg_mask = torch.bitwise_and(neg_mask.bool(), extra_neg_mask.bool()).float()
    else:
        neg_mask = 1. - pos_mask
    return pos_mask, neg_mask

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
        gs = [global_add_pool(z, batch) for z in zs]
        g = torch.cat(gs, dim=1)
        return z, g

class MVGRLEncoder(nn.Module):
    def __init__(self, bloom_filter_dim, in_channels, hidden_channels, out_channels, num_layers):
        super(MVGRLEncoder, self).__init__()
        self.codebook = nn.EmbeddingBag(bloom_filter_dim+1, in_channels, padding_idx=bloom_filter_dim)
        self.mlp1 = MLP(hidden_channels, hidden_channels)
        self.mlp2 = MLP(num_layers * hidden_channels, hidden_channels)
        self.gnn1 = GConv(in_channels, hidden_channels, num_layers)
        self.gnn2 = GConv(in_channels, hidden_channels, num_layers)
        self.aug =  PPRDiffusion(alpha=0.2, use_cache=False)

    def forward(self, x, edge_index, batch):
        x = self.codebook(x)
        x2, edge_index2, edge_weight2 = self.aug(x, edge_index) 
        lv1, gv1 = self.gnn1(x, edge_index, batch)
        lv2, gv2 = self.gnn2(x2, edge_index2, batch)

        lv1 = self.mlp1(lv1)
        lv2 = self.mlp1(lv2)

        gv1 = self.mlp2(gv1)
        gv2 = self.mlp2(gv2)

        return lv1, gv1, lv2, gv2

    def embed(self, x, edge_index, batch):
        _, gv1, _, gv2 = self.forward(x, edge_index, batch)
        return (gv1 + gv2).detach()

class DualBranchContrast(torch.nn.Module):
    def __init__(self, loss, mode: str, intraview_negs: bool = False, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.loss = loss
        self.mode = mode
        self.sampler = get_sampler(mode, intraview_negs=intraview_negs)
        self.kwargs = kwargs

    def forward(self, h1=None, h2=None, g1=None, g2=None, batch=None, h3=None, h4=None,
                extra_pos_mask=None, extra_neg_mask=None):
        if self.mode == 'L2L':
            assert h1 is not None and h2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=h1, sample=h2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=h2, sample=h1)
        elif self.mode == 'G2G':
            assert g1 is not None and g2 is not None
            anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=g2)
            anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=g1)
        else:  # global-to-local
            if batch is None or batch.max().item() + 1 <= 1:  # single graph
                assert all(v is not None for v in [h1, h2, g1, g2, h3, h4])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, neg_sample=h4)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, neg_sample=h3)
            else:  # multiple graphs
                assert all(v is not None for v in [h1, h2, g1, g2, batch])
                anchor1, sample1, pos_mask1, neg_mask1 = self.sampler(anchor=g1, sample=h2, batch=batch)
                anchor2, sample2, pos_mask2, neg_mask2 = self.sampler(anchor=g2, sample=h1, batch=batch)

        pos_mask1, neg_mask1 = add_extra_mask(pos_mask1, neg_mask1, extra_pos_mask, extra_neg_mask)
        pos_mask2, neg_mask2 = add_extra_mask(pos_mask2, neg_mask2, extra_pos_mask, extra_neg_mask)
        l1 = self.loss(anchor=anchor1, sample=sample1, pos_mask=pos_mask1, neg_mask=neg_mask1, **self.kwargs)
        l2 = self.loss(anchor=anchor2, sample=sample2, pos_mask=pos_mask2, neg_mask=neg_mask2, **self.kwargs)

        return (l1 + l2) * 0.5


def train_mvgrl(model, contrast_model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        x = torch.index_select(poi_features, 0, data.x).to_dense().int()
        lv1, gv1, lv2, gv2 = model(x, data.edge_index, data.batch)
        loss = contrast_model(h1=lv1, h2=lv2, g1=gv1, g2=gv2, batch=data.batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def test_mvgrl(model, data_loader):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to('cuda')
            x = torch.index_select(poi_features, 0, data.x).to_dense().int()
            graph_embedding = model.embed(x, data.edge_index, data.batch)
            all_pred.append(graph_embedding.cpu().numpy())
    return np.concatenate(all_pred, axis=0)

def train_mvgrl_and_generate_embeddings(run_id, train_loader, test_loader):
    # Initialize the MVGRL model
    encoder = MVGRLEncoder(8192, in_channels, hidden_channels, out_channels, num_layers).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    contrast_model = DualBranchContrast(loss=JSD(), mode='G2L').to('cuda')

    # Train MVGRL
    pbar = tqdm(range(50), desc=f'Run {run_id+1}', unit='epoch')
    for epoch in pbar:
        loss = train_mvgrl(encoder, contrast_model, train_loader, optimizer)
        pbar.set_postfix({
            'Train Loss': f'{loss:.4f}'
        })
    pbar.close()

    # Generate embeddings
    embeddings = test_mvgrl(encoder, test_loader)
    return embeddings


if __name__ == '__main__':
    total_runs = 3 # ablations doesn't need many runs
    task_results = {
        'population': [],
        'house_price': []
    }
    population_ids, population_labels = extract_ids_and_labels(regions, 'population')
    house_price_ids, house_price_labels = extract_ids_and_labels(regions, 'house_price')
    for i in range(total_runs):
        # Train DGI and generate embeddings
        # Create a single DataLoader for all graphs
        train_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(graphs, batch_size=args.batch_size, shuffle=False)
        run_embeddings = train_mvgrl_and_generate_embeddings(i, train_loader, test_loader)
        # Evaluate using 5-fold cross-validation
        population_metrics = evaluate_rf_cv(run_embeddings[population_ids], population_labels, n_splits=5, random_state=i)
        house_price_metrics = evaluate_rf_cv(run_embeddings[house_price_ids], house_price_labels, n_splits=5, random_state=i)
        task_results['population'].append(population_metrics)
        task_results['house_price'].append(house_price_metrics)
        print(f'# Task: Population Prediction, MAE: {population_metrics["MAE"]:.4f}, RMSE: {population_metrics["RMSE"]:.4f}, R2: {population_metrics["R2"]:.4f}')
        print(f'# Task: House Price Prediction, MAE: {house_price_metrics["MAE"]:.4f}, RMSE: {house_price_metrics["RMSE"]:.4f}, R2: {house_price_metrics["R2"]:.4f}')
        
    avg_results = {task: {metric: mean([result[metric] for result in task_results[task]]) for metric in task_results[task][0]} for task in task_results}
    std_results = {task: {metric: stdev([result[metric] for result in task_results[task]]) for metric in task_results[task][0]} for task in task_results}

    for task_name in task_results:
        header = f"# {'Metric':<10} {'Average':<10} {'Std Dev':<10}"
        separator = '# ' + '-' * len(header)
        print(f'# Task: {task_name} Prediction, City: {dataset}')
        print(header)
        print(separator)
        for metric in avg_results[task_name]:
            avg_value = avg_results[task_name][metric]
            std_value = std_results[task_name][metric]
            print(f"# {metric:<10} {avg_value:.4f} ± {std_value:.4f}")


'''
MVGRL with Bloom Filter inputs
# Task: population Prediction, City: Beijing
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        6162.2521 ± 255.2518
# RMSE       8943.6356 ± 372.2643
# R2         0.3974 ± 0.0479
# Task: house_price Prediction, City: Beijing
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        15257.2495 ± 488.3684
# RMSE       21918.8165 ± 758.3303
# R2         0.7085 ± 0.0238

# Task: population Prediction, City: Shanghai
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        10294.4318 ± 181.4954
# RMSE       14315.1838 ± 298.8510
# R2         0.4416 ± 0.0215
# Task: house_price Prediction, City: Shanghai
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        17344.6675 ± 613.0060
# RMSE       24841.1987 ± 436.8328
# R2         0.2130 ± 0.0103
'''