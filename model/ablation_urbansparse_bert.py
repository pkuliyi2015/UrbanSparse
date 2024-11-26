# This script tests unsupervised Multi-View Graph Representation Learning, with the Bloom filters as the input features.
import argparse
import torch
import numpy as np
import torch.utils.data
import torch_geometric

from torch import nn
from collections import OrderedDict
from tqdm import tqdm
from tqdm.auto import trange
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.spatial import Delaunay
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statistics import mean, stdev
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold

from model.sampler import drop_node
from model.utils import load_region_data
from model.dataset import POIDataset
from model.model_urbansparse import RegionInfoContrast
from model.loss import JSD

argparser = argparse.ArgumentParser()
argparser.add_argument('--dataset', type=str, default='Beijing', help='city name')
argparser.add_argument('--batch_size', type=int, default=16, help='batch size')
argparser.add_argument('--runs', type=int, default=5, help='number of runs')
args = argparser.parse_args()
dataset = args.dataset


poi_dataset = POIDataset(dataset, num_slices=2, num_bits=4096, load_query=False)
poi_locations = poi_dataset.poi_locs
poi_features = np.load(f'baselines/BERT/embeddings/{dataset}/poi_embeddings.npy')
poi_features = torch.tensor(poi_features, dtype=torch.float32).cuda()
region_data = load_region_data(dataset)

region_data = OrderedDict(sorted(region_data.items()))
# The region data is a dict of dicts.
# The outer dict is keyed by the region id.
# The inner dict is keyed by the 'pois', 'population', 'house_price'
# The value of 'pois' is a list of dicts. Each dict contains the 'index' and 'location' of a POI.
# We now count the total number of POIs in the region data.
poi_region_map = {}
region_poi_map = {}
total_pois = 0
for idx, region_info in enumerate(region_data.values()):
    total_pois += len(region_info['pois'])
    region_poi_list= []
    for poi in region_info['pois']:
        poi_id = poi['index']
        poi_region_map[poi_id] = idx
        region_poi_list.append(poi_id)
    region_poi_map[idx] = region_poi_list

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

class RegionData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index_local':
            return self.x_local.size(0)
        if key == 'edge_index_global':
            return self.x_global.size(0)
        return super().__inc__(key, value, *args, **kwargs)

def create_subgraph_from_region(region_info, large_graph):
    if not region_info['pois']:
        raise ValueError("No POIs found in the region")
    
    poi_indices = [poi['index'] for poi in region_info['pois']]
    
    # Control the sampling order of the subgraph according to the number of POIs in the region
    # This is to ensure that the subgraph is not too large

    if len(poi_indices) > 800:
        # in such a large region, we don't need to sample the local region
        local_subset = poi_indices
        local_edge_index = torch_geometric.utils.subgraph(local_subset, large_graph.edge_index, relabel_nodes=True)[0]
        global_subset, global_edge_index, global_mapping, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=poi_indices, num_hops=2, edge_index=large_graph.edge_index, relabel_nodes=True, directed=True
        )
    else:
        # local view
        local_subset, local_edge_index, local_mapping, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=poi_indices, num_hops=2, edge_index=large_graph.edge_index, relabel_nodes=True
        )
        global_subset, global_edge_index, global_mapping, _ = torch_geometric.utils.k_hop_subgraph(
            node_idx=poi_indices, num_hops=4, edge_index=large_graph.edge_index, relabel_nodes=True, directed=True
        )

    # global
    # Select the features of the nodes in the subset from large_graph.x
    local_x = large_graph.x[local_subset]
    global_x = large_graph.x[global_subset]

    return RegionData(
        x_local=local_x,
        edge_index_local=local_edge_index,
        x_global=global_x,
        edge_index_global=global_edge_index
    )

# Create subgraphs for each region
graphs = []
regions = []
for region_id, region_info in tqdm(region_data.items(), desc='Creating Subgraphs'):
    if region_info['pois']:
        graphs.append(create_subgraph_from_region(region_info, large_graph))
        regions.append(region_info)
print(f'Created {len(graphs)} subgraphs')


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
        # add them together
        g = torch.cat(gs, dim=1)
        return z, g

class UrbanSparse(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        '''
        We failed to design a good retrieval model on BERT embeddings.
        Hence we only ablation the region contrastive learning part.
        '''
        super(UrbanSparse, self).__init__()
        self.encoder = nn.Linear(in_channels, hidden_channels, bias=False)
        self.mlp1 = MLP(hidden_channels, hidden_channels)
        self.mlp2 = MLP(2 * hidden_channels, hidden_channels)
        self.gnn1 = GConv(hidden_channels, hidden_channels, num_layers)
        self.gnn2 = GConv(hidden_channels, hidden_channels, num_layers)
        self.activation = nn.PReLU()

    def region_contrast(self, local_x: torch.LongTensor, global_x: torch.LongTensor, local_edge_index: torch.LongTensor, global_edge_index: torch.LongTensor, local_batch: torch.LongTensor, global_batch: torch.LongTensor):
        
        local_x_emb = self.encoder(local_x)
        global_x_emb = self.encoder(global_x)
        # activation to increase the non-linearity
        local_x_emb = self.activation(local_x_emb)
        global_x_emb = self.activation(global_x_emb)
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
    


def train_model(model, contrast_loss, data_loader, optimizer):
    model.train()
    total_loss = 0
    for data in data_loader:
        data = data.to('cuda')
        optimizer.zero_grad()
        x_local = torch.index_select(poi_features, 0, data.x_local).float()
        x_global = torch.index_select(poi_features, 0, data.x_global).float()
        lv1, gv1, lv2, gv2 = model.region_contrast(x_local, x_global, data.edge_index_local, data.edge_index_global, data.x_local_batch, data.x_global_batch)
        loss = contrast_loss(h1=lv1, h2=lv2, g1=gv1, g2=gv2, h1_batch=data.x_local_batch, h2_batch=data.x_global_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(data_loader)

def test_model(model, data_loader):
    model.eval()
    all_pred = []
    with torch.no_grad():
        for data in data_loader:
            data = data.to('cuda')
            x_local = torch.index_select(poi_features, 0, data.x_local).float()
            x_global = torch.index_select(poi_features, 0, data.x_global).float()
            lv1, gv1, lv2, gv2 = model.region_contrast(x_local, x_global, data.edge_index_local, data.edge_index_global, data.x_local_batch, data.x_global_batch)
            region_embeddings = torch.cat([gv1, gv2], dim=0)
            all_pred.append(region_embeddings.cpu().numpy())
    return np.concatenate(all_pred, axis=0)

def train_model_and_generate_embeddings(run_id, train_loader, test_loader):
    # Initialize the MVGRL model
    # Initialize the MVGRL model
    in_channels = poi_features.shape[1]
    hidden_channels = 64
    out_channels = 64
    num_layers = 2

    encoder = UrbanSparse(in_channels, hidden_channels, out_channels, num_layers).to('cuda')
    optimizer = torch.optim.Adam(encoder.parameters(), lr=0.001)
    contrast_model = RegionInfoContrast(loss=JSD(), mode='G2L').to('cuda')

    # Train MVGRL
    pbar = tqdm(range(100), desc=f'Run {run_id+1}', unit='epoch')
    for epoch in pbar:
        loss = train_model(encoder, contrast_model, train_loader, optimizer)
        pbar.set_postfix({
            'Train Loss': f'{loss:.4f}'
        })
    pbar.close()

    # Generate embeddings
    embeddings = test_model(encoder, test_loader)
    return embeddings

def evaluate_rf_cv(embeddings, labels, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mae_scores, rmse_scores, r2_scores = [], [], []
    
    for train_index, test_index in tqdm(kf.split(embeddings), desc='Evaluating RF with k-fold CV', total=n_splits):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2_scores.append(r2_score(y_test, y_pred))
    
    return {
        'MAE': np.mean(mae_scores),
        'RMSE': np.mean(rmse_scores),
        'R2': np.mean(r2_scores)
    }

def extract_ids_and_labels(regions, label_key):
    labels = []
    ids = []
    for idx, region_info in enumerate(regions):
        if region_info[label_key] > 0: # only consider regions with valid ground truth
            labels.append(region_info[label_key])
            ids.append(idx)
    return ids, np.array(labels)

if __name__ == '__main__':
    total_runs = args.runs
    task_results = {
        'population': [],
        'house_price': []
    }
    population_ids, population_labels = extract_ids_and_labels(regions, 'population')
    house_price_ids, house_price_labels = extract_ids_and_labels(regions, 'house_price')
    for i in range(total_runs):
        # Train MVGRL and generate embeddings
        # Create a single DataLoader for all graphs
        train_loader = DataLoader(graphs, batch_size=32, shuffle=True, follow_batch=['x_local', 'x_global'])
        test_loader = DataLoader(graphs, batch_size=64, shuffle=False, follow_batch=['x_local', 'x_global'])
        run_embeddings = train_model_and_generate_embeddings(i, train_loader, test_loader)
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
UrbanSparse on BERT embeddings
# Task: population Prediction, City: Beijing
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        6176.9563 ± 111.4767
# RMSE       9035.9402 ± 130.3340
# R2         0.3830 ± 0.0172
# Task: house_price Prediction, City: Beijing
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        23398.3643 ± 337.9413
# RMSE       33598.8081 ± 492.7353
# R2         0.3388 ± 0.0252

# Task: population Prediction, City: Shanghai
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        11501.9884 ± 180.1305
# RMSE       15975.8744 ± 47.0212
# R2         0.3046 ± 0.0068
# Task: house_price Prediction, City: Shanghai
# Metric     Average    Std Dev   
# ----------------------------------
# MAE        19148.8883 ± 335.0122
# RMSE       26567.6341 ± 375.8543
# R2         0.1010 ± 0.0323
'''