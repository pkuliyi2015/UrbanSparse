import os
import argparse
import time
import torch
import numpy as np
import torch_geometric

from scipy.spatial import Delaunay
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.data import Data as GeometricData
from torch_geometric.loader import DataLoader as GeometricDataLoader

from model.utils import eval_search, load_region_data, extract_ids_and_labels, evaluate_rf_cv
from model.dataset import POIDataset, POIRetrievalDataset
from model.model_urbansparse import UrbanSparse, RegionInfoContrast
from model.loss import lambdaLoss, JSD

@torch.no_grad()
def generate_negative_samples(model, query_dataset, poi_bloom_filter, poi_weights, poi_locs, batch_size=64):
    negative_samples = []
    query_dataloader = TorchDataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=query_dataset.collate_fn)
    for data in tqdm(query_dataloader, desc="Generating Negative Samples via Brute Force Search"):
        query_bloom_filter, query_loc, query_truth, query_weights = data
        query_bloom_filter = query_bloom_filter.cuda().to(torch.int32)
        query_bloom_filter[query_bloom_filter == -1] = padding_idx
        query_loc = query_loc.cuda()
        query_weights = query_weights.cuda()
        relevance_score = model.brute_force_search(query_bloom_filter, query_weights, query_loc, poi_bloom_filter, poi_weights, poi_locs)
        top_k_indices = torch.topk(relevance_score, 400, dim=-1, largest=True)[1]
        negative_samples.append(top_k_indices.cpu())
    return torch.cat(negative_samples, dim=0).numpy()

@torch.no_grad()
def evaluate_retrieval(model, query_dataset, poi_bloom_filter, poi_locs, batch_size=16):
    query_dataloader = TorchDataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=query_dataset.collate_fn)
    query_weights = []
    for data in query_dataloader:
        query_bloom_filter, _, _, _ = data
        query_bloom_filter = query_bloom_filter.cuda().to(torch.int32)
        query_bloom_filter[query_bloom_filter == -1] = padding_idx
        query_weights.append(model.encode_query(query_bloom_filter))
    query_weights = torch.vstack(query_weights)
    poi_weights = model.encode_poi(poi_bloom_filter)

    query_dataset.set_weights(query_weights.cpu())
    query_dataloader = TorchDataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=query_dataset.collate_fn)
    relevance_scores = []
    query_truths = []
    start_time = time.time()
    for data in tqdm(query_dataloader, desc="Evaluating Retrieval"):
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
    return eval_results

@torch.no_grad()
def refresh_train_negs(model, query_dataset, poi_bloom_filter, poi_locs, batch_size=128):
    query_dataloader = TorchDataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=query_dataset.collate_fn)
    query_weights = []
    for data in query_dataloader:
        query_bloom_filter, _, _, _ = data
        query_bloom_filter = query_bloom_filter.cuda().to(torch.int32)
        query_bloom_filter[query_bloom_filter == -1] = padding_idx
        query_weights.append(model.encode_query(query_bloom_filter))
    query_weights = torch.vstack(query_weights)
    poi_weights = model.encode_poi(poi_bloom_filter)

    query_dataset.set_weights(query_weights.cpu())
    query_dataloader = TorchDataLoader(query_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, collate_fn=query_dataset.collate_fn)
    top_k_indices = []
    for data in tqdm(query_dataloader, desc="Evaluating Retrieval"):
        query_bloom_filter, query_loc, query_truth, query_weights = data
        query_bloom_filter = query_bloom_filter.cuda().to(torch.int32)
        query_bloom_filter[query_bloom_filter == -1] = padding_idx
        query_loc = query_loc.cuda()
        query_weights = query_weights.cuda()
        relevance_score = model.brute_force_search(query_bloom_filter, query_weights, query_loc, poi_bloom_filter, poi_weights, poi_locs)
        top_k_indices.append(torch.topk(relevance_score, 100, dim=-1, largest=True)[1])
    return torch.cat(top_k_indices, dim=0).cpu().numpy()
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Beijing')
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--refresh', type=int, default=5)
    parser.add_argument('--k-hop', type=int, default=4)
    parser.add_argument('--no-ret', action='store_true')
    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()
    dataset = args.dataset
    batch_size = args.batch_size
    num_epochs = args.epochs
    warmup_epochs = args.warmup
    refresh_epochs = args.refresh
    k_hop = args.k_hop
    no_ret = args.no_ret
    no_pred = args.no_pred
    num_slices = 2
    num_bits = 4096
    padding_idx = num_slices * num_bits
    poi_dataset = POIDataset(dataset, num_slices=num_slices, num_bits=num_bits, load_query=True)
    model = UrbanSparse(bloom_filter_dim=num_slices * num_bits, hidden_1=512, hidden_2=64, dense_dim=64, no_ret=no_ret).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    retrieval_loss = lambda pred, truth: lambdaLoss(pred, truth, k=30, weighing_scheme='lambdaRank_scheme', reduction="sum")
    contrast_loss = RegionInfoContrast(loss=JSD(), mode='G2L').to('cuda')

    query_dataloader = TorchDataLoader(poi_dataset.train_dataset, batch_size=16, shuffle=False, num_workers=2, collate_fn=poi_dataset.train_dataset.collate_fn)
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

    # Preparing Retrieval Negative Samples
    cache_dir = f'model/cache/{dataset}'
    os.makedirs(cache_dir, exist_ok=True) 
    train_neg_path = f'{cache_dir}/train_neg.npy'
    if not os.path.exists(train_neg_path):
        train_neg = generate_negative_samples(model, poi_dataset.train_dataset, poi_bloom_filter, poi_weights, poi_locs)
        np.save(train_neg_path, train_neg)
    else:
        train_neg = np.load(train_neg_path)

    retrieval_dataset = POIRetrievalDataset(poi_dataset.train_dataset.query_bloom_filters, poi_dataset.train_dataset.query_locs, poi_bloom_filter, poi_locs, poi_dataset.train_dataset.truths, train_neg, top_k=200)
    retrieval_dataloader = TorchDataLoader(retrieval_dataset, batch_size=batch_size, shuffle=True, num_workers=8, collate_fn=retrieval_dataset.collate_fn)

    # Preparing Region Data
    region_data = load_region_data(dataset)
    # Change it to an OrderedDict, sort by the key
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
    poi_locations = poi_locs.cpu().numpy()
    if poi_locations.shape[1] != 2:
        raise ValueError("POI locations must be 2D for Delaunay triangulation")

    regions = []
    for region_id, region_info in tqdm(region_data.items(), desc='Creating Subgraphs'):
        if region_info['pois']:
            regions.append(region_info)

    tri = Delaunay(poi_locations)
    edges = set()
    for simplex in tri.simplices:
        edges.update((simplex[i], simplex[j]) for i in range(3) for j in range(i + 1, 3))
        edges.update((simplex[j], simplex[i]) for i in range(3) for j in range(i + 1, 3))
    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    # Create the large graph
    # The x is an index vector of the poi_bloom_filter as the bloom filter is too large to fit in memory
    large_graph = GeometricData(x=poi_bloom_filter, edge_index=edge_index)

    class RegionData(GeometricData):
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
        if k_hop == 0:
            smaller_k_hop = 0
        if k_hop == 1:
            smaller_k_hop = 0
        else:
            smaller_k_hop = k_hop - 1

        if len(poi_indices) > 800:
            # in such a large region, we don't need to sample the local region
            local_subset = poi_indices
            local_edge_index = torch_geometric.utils.subgraph(local_subset, large_graph.edge_index, relabel_nodes=True)[0]
            global_subset, global_edge_index, global_mapping, _ = torch_geometric.utils.k_hop_subgraph(
                node_idx=poi_indices, num_hops=k_hop, edge_index=large_graph.edge_index, relabel_nodes=True, directed=True
            )
        else:
            # local view
            local_subset, local_edge_index, local_mapping, _ = torch_geometric.utils.k_hop_subgraph(
                node_idx=poi_indices, num_hops=smaller_k_hop, edge_index=large_graph.edge_index, relabel_nodes=True
            )
            global_subset, global_edge_index, global_mapping, _ = torch_geometric.utils.k_hop_subgraph(
                node_idx=poi_indices, num_hops=k_hop, edge_index=large_graph.edge_index, relabel_nodes=True, directed=True
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
    for i, region_info in enumerate(regions):
        if region_info['pois']:
            graphs.append(create_subgraph_from_region(region_info, large_graph))



    prediction_train_loader = GeometricDataLoader(graphs, batch_size=32, shuffle=True, follow_batch=['x_local', 'x_global'])
    prediction_test_loader = GeometricDataLoader(graphs, batch_size=64, shuffle=False, follow_batch=['x_local', 'x_global'])

    population_ids, population_labels = extract_ids_and_labels(regions, 'population')
    house_price_ids, house_price_labels = extract_ids_and_labels(regions, 'house_price')


    iter_dl1 = iter(retrieval_dataloader)
    iter_dl2 = iter(prediction_train_loader)

    model_path = f'model/cache/{dataset}/urbansparse'
    if no_pred:
        model_path += '_no_pred'
    model_path += '.pth'

    # Evaluate the model when no training is performed (i.e. UrbanSparse w/o Individual View)
    if not no_ret:
        model.eval()
        with torch.no_grad():
            test_results = evaluate_retrieval(model, poi_dataset.test_dataset, poi_bloom_filter, poi_locs, batch_size=64)
            print(f'Test Results: {test_results}')

    # Training loop
    best_dev_ndcg_5 = 0
    for epoch in range(num_epochs):
        # retrieve = (not no_ret and epoch > warmup_epochs) or no_pred
        # ablation: no warmup
        retrieve = not no_ret
        train_relevance_scores = []
        train_candidate_truths = []
        model.train()

        retrieval_loss_list = []
        prediction_loss_list = []

        # Warm-up the codebook via region-level contrastive learning
        pbar = tqdm(retrieval_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        p_loss, r_loss = None, None
        for batch in pbar:
            if not no_pred:
                # We do more training on the prediction task to get better codebook.
                for _ in range(2):
                    try:
                        batch2 = next(iter_dl2)
                        batch2 = batch2.to('cuda')
                    except StopIteration:
                        iter_dl2 = iter(prediction_train_loader)  # Reset dataloader2 if it ends
                        batch2 = next(iter_dl2)
                        batch2 = batch2.to('cuda')
                    optimizer.zero_grad()
                    lv1, gv1, lv2, gv2 = model.region_contrast(batch2.x_local, batch2.x_global, batch2.edge_index_local, batch2.edge_index_global, batch2.x_local_batch, batch2.x_global_batch)
                    p_loss = contrast_loss(h1=lv1, h2=lv2, g1=gv1, g2=gv2, h1_batch=batch2.x_local_batch, h2_batch=batch2.x_global_batch)
                    p_loss.backward()
                    optimizer.step()
                    prediction_loss = p_loss.item()
                    prediction_loss_list.append(prediction_loss)

            # Start training on retrieval task after the warm-up phase
            if retrieve:
                optimizer.zero_grad()
                query, query_loc, candidate_pois, candidate_poi_locs, candidate_truths = batch
                query = query.to('cuda', dtype=torch.int32)
                query[query == -1] = padding_idx
                query_loc = query_loc.cuda()
                candidate_pois = candidate_pois.to('cuda', dtype=torch.int32)
                candidate_poi_locs = candidate_poi_locs.cuda()
                candidate_truths = candidate_truths.cuda()
                # Forward pass
                relevance_scores = model.bi_encode(query, query_loc, candidate_pois, candidate_poi_locs)
                # Compute loss
                r_loss = retrieval_loss(relevance_scores, candidate_truths)
                # Backward pass and optimization
                r_loss.backward()
                optimizer.step()
                retrieval_loss_list.append(r_loss.item())

            if p_loss is not None and r_loss is not None:
                pbar.set_postfix(retrieval_loss=f'{r_loss.item():.4f}', prediction_loss=f'{p_loss.item():.4f}')
            elif r_loss is not None:
                pbar.set_postfix(retrieval_loss=f'{r_loss.item():.4f}')
            elif p_loss is not None:
                pbar.set_postfix(prediction_loss=f'{p_loss.item():.4f}')

        # Print the average losses for the epoch
        print(f'Epoch {epoch+1}/{num_epochs} - Retrieval Loss: {np.mean(retrieval_loss_list)}, Prediction Loss: {np.mean(prediction_loss_list)}')

        # Brute force search on dev set
        model.eval()
        if retrieve:
            dev_results = evaluate_retrieval(model, poi_dataset.dev_dataset, poi_bloom_filter, poi_locs, batch_size=16)
            print(f'Dev Results: {dev_results}')
            search_ndcg_5 = dev_results['NDCG @ 5']
            if search_ndcg_5 > best_dev_ndcg_5:
                best_dev_ndcg_5 = search_ndcg_5
                torch.save(model.state_dict(), model_path)
                print(f'New best dev NDCG@5: {best_dev_ndcg_5}. Model saved to {model_path}')

        if not no_pred:
            print('Evaluating graph embeddings')
            with torch.no_grad():
                test_embeddings = []
                for batch in tqdm(prediction_test_loader, desc='Extracting test embeddings'):
                    batch = batch.to('cuda')
                    _, region_embedding, _, context_embedding = model.region_contrast(batch.x_local, batch.x_global, batch.edge_index_local, batch.edge_index_global, batch.x_local_batch, batch.x_global_batch)
                    test_embedding =  torch.cat([region_embedding, context_embedding], dim=-1)
                    test_embeddings.append(test_embedding.cpu().numpy())
                test_embeddings = np.concatenate(test_embeddings, axis=0)
                population_metrics = evaluate_rf_cv(test_embeddings[population_ids], population_labels, n_splits=5, random_state=0)
                house_price_metrics = evaluate_rf_cv(test_embeddings[house_price_ids], house_price_labels, n_splits=5, random_state=0)
                print(population_metrics)
                print(house_price_metrics)


    # load the best model
    model.load_state_dict(torch.load(model_path))

    # evaluate the model
    model.eval()
    with torch.no_grad():
        eval_log_path = f'logs/{dataset}'
        if no_pred:
            eval_log_path += '_no_pred'
        if no_ret:
            eval_log_path += '_no_ret'
        eval_log_path += '.txt'
        file = open(eval_log_path, 'a+')
        if not no_ret:
            test_results = evaluate_retrieval(model, poi_dataset.test_dataset, poi_bloom_filter, poi_locs, batch_size=64)
            print(f'Test Results: {test_results}')
            # the test results are in {metric: np.float64}
            file.write('Retrieval:\t')
            for metric, value in test_results.items():
                file.write(f'{metric}: {value}\t')
            file.write('\n')
        if not no_pred:
            test_embeddings = []
            for batch in tqdm(prediction_test_loader, desc='Extracting test embeddings'):
                batch = batch.to('cuda')
                _, region_embedding, _, context_embedding = model.region_contrast(batch.x_local, batch.x_global, batch.edge_index_local, batch.edge_index_global, batch.x_local_batch, batch.x_global_batch)
                test_embedding =  torch.cat([region_embedding, context_embedding], dim=-1)
                test_embeddings.append(test_embedding.cpu().numpy())
            test_embeddings = np.concatenate(test_embeddings, axis=0)
            population_metrics = evaluate_rf_cv(test_embeddings[population_ids], population_labels, n_splits=5, random_state=0)
            house_price_metrics = evaluate_rf_cv(test_embeddings[house_price_ids], house_price_labels, n_splits=5, random_state=0)
            print(population_metrics)
            print(house_price_metrics)
            file.write('Population Prediction:\t')
            for metric, value in population_metrics.items():
                file.write(f'{metric}: {value}\t')
            file.write('\n')
            file.write('House Price Prediction:\t')
            for metric, value in house_price_metrics.items():
                file.write(f'{metric}: {value}\t')
            file.write('\n')
        file.close()

'''

UrbanSparse on Bloom filters
# Task: population Prediction, City: Beijing
# Metric     Average       Std Dev   
# --------------------------------------
# MAE        3307.91       ± 14.29
# RMSE       5772.70       ± 31.76
# R2         0.75          ± 0.003

# Task: house_price Prediction, City: Beijing
# Metric     Average       Std Dev   
# --------------------------------------
# MAE        11983.21      ± 506.79
# RMSE       17155.12      ± 766.57
# R2         0.82          ± 0.013

# Task: population Prediction, City: Shanghai
# Metric     Average       Std Dev   
# --------------------------------------
# MAE        5343.72       ± 55.47
# RMSE       8958.22       ± 169.41
# R2         0.78          ± 0.01

# Task: house_price Prediction, City: Shanghai
# Metric     Average       Std Dev   
# --------------------------------------
# MAE        13280.88      ± 325.13
# RMSE       20609.66      ± 671.06
# R2         0.46          ± 0.04


DPR QPS
Beijing 133.05
Shanghai 133.94

UrbanSparse QPS
Beijing 476.2851740847886
Shanghai 505.19710203271734
'''