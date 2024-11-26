import numpy as np
import pickle as pkl
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

def fast_ndcg(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    if len(truth) == 0:
        return 0
    # remove -1 from truth
    truth = [t for t in truth if t != -1]
    dcg = 0
    for i in range(min(len(prediction), k)):
        if prediction[i] in truth:
            dcg += 1 / np.log2(i + 2)
    idcg = 0
    for i in range(min(len(truth), k)):
        idcg += 1 / np.log2(i + 2)
    return dcg / idcg

def recall(prediction, truth, k):
    # prediction: list of indices
    # truth: list of indices
    # k: int
    # return: float
    if len(truth) == 0:
        return 0
        # remove -1 from truth
    truth = [t for t in truth if t != -1]
    return len(set(prediction[:k]).intersection(set(truth))) / min(len(set(truth)), k)

def eval_search(top_k_indices, query_truth, metrics=None):
    if metrics is None:
        metrics = {
            'Recall @ 20': lambda p, t: recall(p, t, 20),
            'Recall @ 10': lambda p, t: recall(p, t, 10),
            'NDCG @ 5': lambda p, t: fast_ndcg(p, t, 5),
            'NDCG @ 1': lambda p, t: fast_ndcg(p, t, 1),
        }

    results = {}
    for metric_name, metric_func in metrics.items():
        metric_value = np.mean([metric_func(p, t) for p, t in zip(top_k_indices, query_truth)])
        results[metric_name] = metric_value
    return results


def load_region_data(city):
    if (city == 'Beijing'):
        filepath = 'data/Predictions/BJ_data.pkl'
        with open(filepath, 'rb') as file:
            region_data = pkl.load(file)
    elif (city == 'Shanghai'):
        filepath = 'data/Predictions/SH_data.pkl'
        with open(filepath, 'rb') as file:
            region_data = pkl.load(file)
    else:
        raise ValueError("Unsupported city. Please choose city among 'Beijing' or 'Shanghai'")
    
    return region_data

def evaluate_rf_cv(embeddings, labels, n_splits=5, random_state=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    mae_scores, rmse_scores, r2_scores = [], [], []
    
    for train_index, test_index in tqdm(kf.split(embeddings), desc='Evaluating RF with k-fold CV', total=n_splits):
        X_train, X_test = embeddings[train_index], embeddings[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        
        rf = RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=32)
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