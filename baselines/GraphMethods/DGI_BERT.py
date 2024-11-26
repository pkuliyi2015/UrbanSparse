# This script tests unsupervised Deep Graph Infomax, with the Bloom filters as the input features.
import argparse
import torch
import numpy as np
import torch.utils.data
import torch_geometric

from torch import nn
from tqdm import tqdm
from torch_geometric.nn import GCN
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import Delaunay
from torch_geometric.nn import global_mean_pool

from model.dataset import POIDataset
from model.utils import load_region_data, evaluate_rf_cv, extract_ids_and_labels
from statistics import mean, stdev

from torch_geometric.nn import DeepGraphInfomax

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='Beijing', help='city name')
argparser.add_argument('--batch_size', type=int, default=16, help='batch size')
args = argparser.parse_args()
dataset = args.dataset


poi_dataset = POIDataset(dataset, num_slices=2, num_bits=4096, load_query=False)
poi_locations = poi_dataset.poi_locs
poi_features = np.load(f'baselines/BERT/embeddings/{dataset}/poi_embeddings.npy')
poi_features = torch.tensor(poi_features, dtype=torch.float32).cuda()
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
Standard DGI from pytorch-geometric with essential modifications.
1. Add 1 layer of GCN.
2. adapted to batch processing.
'''


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(Encoder, self).__init__()
        self.gcn = GCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            num_layers=2,
            out_channels=out_channels
        )

    def forward(self, x, edge_index, batch):
        # The GCN layer does not need batch information as node accross batches are not connected.
        return self.gcn(x, edge_index)

def corruption(x, edge_index, batch):
    return x[torch.randperm(x.size(0))], edge_index, batch

def summary(z, x, edge_index, batch, *args, **kwargs):
    '''
    The summary function requires the batch information to perform per-graph pooling.
    '''
    return global_mean_pool(z, batch)

# Initialize the DGI model. Default settings 
in_channels = poi_features.shape[1]
hidden_channels = 512
out_channels = 512

def train_dgi(model, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        x = torch.index_select(poi_features, 0, data.x).to_dense().float()
        pos_z, neg_z, summary = model(x, data.edge_index, data.batch)
        loss = model.loss(pos_z, neg_z, summary)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def test_dgi(model, data_loader):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to('cuda')
            x = torch.index_select(poi_features, 0, data.x).to_dense().float()
            z, _, _ = model(x, data.edge_index, data.batch)
            graph_embedding = global_mean_pool(z, data.batch)
            all_pred.append(graph_embedding.cpu().numpy())
    return np.concatenate(all_pred, axis=0)

def train_dgi_and_generate_embeddings(run_id, train_loader, test_loader):
    # Initialize the DGI model
    encoder = Encoder(in_channels, hidden_channels, out_channels).to('cuda')
    dgi_model = DeepGraphInfomax(
        hidden_channels=out_channels,
        encoder=encoder,
        summary=summary,
        corruption=corruption
    ).to('cuda')
    optimizer = torch.optim.Adam(dgi_model.parameters(), lr=0.001)

    # Train DGI
    pbar = tqdm(range(50), desc=f'Run {run_id+1}', unit='epoch')
    for epoch in pbar:
        loss = train_dgi(dgi_model, train_loader, optimizer)
        pbar.set_postfix({
            'Train Loss': f'{loss:.4f}'
        })
    pbar.close()

    # Generate embeddings
    embeddings = test_dgi(dgi_model, test_loader)
    return embeddings


if __name__ == '__main__':
    total_runs = 10
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
        run_embeddings = train_dgi_and_generate_embeddings(i, train_loader, test_loader)
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
DGI with BERT inputs
# Task: population Prediction, City: Beijing
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        5777.1923 ± 107.4457
# RMSE       8380.9394 ± 102.8677
# R2         0.4682 ± 0.0138
# Task: house_price Prediction, City: Beijing
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        15319.2726 ± 391.1100
# RMSE       21546.8794 ± 977.2803
# R2         0.7240 ± 0.0289

# Task: population Prediction, City: Shanghai
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        13037.4693 ± 380.4239
# RMSE       17265.0221 ± 279.6464
# R2         0.1877 ± 0.0274
# Task: house_price Prediction, City: Shanghai
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        18374.1678 ± 251.8329
# RMSE       25747.6407 ± 465.2531
# R2         0.1511 ± 0.0318
'''

