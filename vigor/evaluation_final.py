import pandas as pd
import numpy as np
import itertools
import torch
from torch import nn, optim
from vigor.predicate import Predicate

device = "cuda" if torch.cuda.is_available() else "cpu"

def bce(y, balanced=False):
    n_points = len(y)
    n_selected = y.sum()
    n_unselected = n_points - n_selected
    instance_weight = torch.ones(y.shape[0]).to(device)
    if balanced:
        instance_weight *= (y*(n_unselected/n_points) + (1-y)*(n_selected/n_points))
    return nn.BCELoss(weight=instance_weight)

def predict(x, a, mu):
    b = 4
    pred = 1 / (1 + ((a.abs() * (x - mu).abs()).pow(b)).sum(1))
    return pred

def denormalize(x, norm_min, norm_max):
    return x * (norm_max - norm_min) + norm_min

def normalize(x, norm):
    return (x - norm['min'])/(norm['max']-norm['min'])

def interval_to_params(interval):
    return (interval[1] + interval[0])/2, 1 / ((interval[1] - interval[0])/2)

def params_to_interval(mu, a):
    return [mu-1/a, mu+1/a]

def to_numpy(tensor):
    return tensor.detach().numpy()

def get_metrics(y, pred, metrics):
    metric_columns = [a for b in [[k] if type(k)==str else k for k in metrics.keys()] for a in b]
    s = [v(y, pred) for v in metrics.values()]
    scores = [a for b in [[si] if type(si)==float else si for si in s] for a in b]
    return pd.Series(scores, index=metric_columns)

def intervals_to_df(mu, a, columns, norm):
    intervals_norm = {
        columns[i]: params_to_interval(to_numpy(mu[i]), to_numpy(a[i]))
        for i in range(len(columns))
    }
    intervals = {k: [denormalize(v[0], *norm.loc[k]), denormalize(v[1], *norm.loc[k])] for k,v in intervals_norm.items()}
    intervals_df = pd.concat([
        pd.DataFrame([{f'{k}_min': v[0] for k,v in intervals.items()}]),
        pd.DataFrame([{f'{k}_max': v[1] for k,v in intervals.items()}])
    ], axis=1)
    return intervals_df

def params_to_df(y, pred, mu, a, columns, norm, metrics, loss):
    intervals_df = intervals_to_df(mu, a, columns, norm)
    metrics_df = get_metrics(y, pred, metrics)
    res_df = pd.concat([intervals_df, metrics_df.to_frame().T], axis=1)
    res_df['loss'] = to_numpy(loss)
    return res_df

def get_predicates(graph_data, labels, vistype, predicate_regr, metrics, eps=0.1, **kwargs):
    columns = graph_data.columns.tolist()
    norm = graph_data.agg(['min', 'max']).T
    graph_data_norm = normalize(graph_data, norm)
    kwargs_norm = {k: [normalize(v[0], norm), normalize(v[1], norm)] for k,v in kwargs.items()}
    
    labeled_stats = graph_data_norm[labels].agg(['mean', 'std']).T    
    not_null_columns = labeled_stats[~labeled_stats.isnull().any(axis=1)].index
    mu_series = labeled_stats['mean']
    a_series = 1/(predicate_regr['init_width']*(labeled_stats['std']+eps))
    for k,v in kwargs_norm.items():
        column_index = columns.index(k)
        mu_series[column_index], a_series[column_index] = interval_to_params(v)
    
    not_null = ~graph_data_norm[not_null_columns].isnull().any(axis=1)
    not_null_rows = not_null[not_null].index
    x = torch.from_numpy(graph_data_norm.loc[not_null_rows, not_null_columns].values).to(device)
    y = torch.from_numpy(labels[not_null_rows].astype(float).values).to(device)
    mu = torch.from_numpy(mu_series[not_null_columns].values).to(device)
    a = torch.from_numpy(a_series[not_null_columns].values).to(device)
    a.requires_grad_(True)
    mu.requires_grad_(True)
        
    optimizer = optim.SGD(
            [{"params": mu, "weight_decay": predicate_regr['weight_decay_mu']}, {"params": a, "weight_decay": predicate_regr['weight_decay_a']}],
            **{k:v for k,v in predicate_regr.items() if 'weight_' not in k and k not in ['n_iter', 'balanced', 'init_width']}
        )
    loss = bce(y, balanced=predicate_regr['balanced'])
    res_list = []
    pred_list = []
    for e in range(predicate_regr['n_iter']+1):
        pred = predict(x, a, mu)
        pred_list.append(pred.detach().numpy())
        loss_ = loss(pred, y)
        optimizer.zero_grad()
        loss_.backward()
        optimizer.step()
        res = params_to_df(y, pred, mu, a, not_null_columns, norm, metrics, loss_)
        res_list.append(res)
        optimizer.step()
    return pd.concat(res_list).reset_index(drop=True), pd.DataFrame(pred_list).T

def generate_predicate(data, p, columns=None, k=None):
    if columns is None:
        max_binary = int(.5/p)
        num_binary = np.random.randint(0, max_binary)
        num_numeric = k - num_binary
        binary = np.random.choice(data.binary, size=num_binary).tolist()
        numeric = np.random.choice(data.numeric, size=num_numeric).tolist()
    else:
        binary = [col for col in columns if col in data.binary]
        numeric = [col for col in columns if col in data.numeric]
        num_binary = len(binary)
        num_numeric = len(numeric)

    binary_p = .5**num_binary
    numeric_p = (p/binary_p)**(1/num_numeric) if num_numeric>0 else 1
    numeric_ranges = {col: data.column_dtypes[col].range*numeric_p for col in numeric}
    
    left = {k: np.random.uniform(data.column_dtypes[k].minval, data.column_dtypes[k].maxval-v) for k,v in numeric_ranges.items()}
    intervals = {k: [v, v+numeric_ranges[k]] for k,v in left.items()}
    bin_left = {k: np.random.randint(0, 1) for k in binary}
    bin_intervals = {k: [v, v] for k,v in bin_left.items()}
    clauses = {k: v for k,v in intervals.items()}
    for k,v in bin_intervals.items():
        clauses[k] = v   
    return Predicate(clauses=clauses)

def get_labels(predicate, conformance):
    return (predicate.mask*np.random.binomial(1, conformance, size=len(predicate.mask)) +
        (1-predicate.mask)*np.random.binomial(1, 1-conformance, size=len(predicate.mask))
    ).astype(bool)

def test_predicate(data, predicate, vistype, metrics, predicate_regr, conformance=1, attrs=None):
    predicate.fit(data)
    labels = get_labels(predicate, conformance)
    a,b = get_predicates(
        data[data.columns if attrs is None else attrs], labels, vistype,
        predicate_regr, metrics,
    )
    return labels, a, b

def test_vistype_predicates(data, vistype_predicates, predicate_regr, metrics):
    error_res = {}
    pred_res = {}
    labels_res = {}
    
    for vistype,predicate in vistype_predicates.items():
        predicate.fit(data)
        for conformance in [.5, .75, 1]:
            labels, error, pred = test_predicate(data, predicate, vistype, metrics, predicate_regr, conformance)
            error_res[(vistype, conformance)] = error
            pred_res[(vistype, conformance)] = pred
            labels_res[(vistype, conformance)] = labels
    return error_res, pred_res, labels_res

def get_args(all_data, all_predicates, all_params, all_metrics):
    all_data_keys = list(all_data.keys())
    all_predicates_keys = list(all_predicates.keys())
    all_params_keys = range(len(all_params))
    all_metrics_keys = range(len(all_metrics))
    
    args = pd.DataFrame(
        list(itertools.product(all_data_keys, all_predicates_keys, all_params_keys, all_metrics_keys)),
        columns=['data', 'predicates', 'params', 'metrics']
    )
    return args

def get_train_res(train_data, predicates, params, metrics):
    train_error, train_pred, train_labels = test_vistype_predicates(train_data, predicates, params, metrics)
    res = pd.DataFrame(
        {k: v.loc[v.loss.idxmin()] for k,v in train_error.items()}
    ).T.reset_index().rename(columns={'level_0': 'vistype', 'level_1':'conformance'})
    return res

def get_test_res(train_res, test_data, metrics):
    res_list = []
    for i,v in train_res.iterrows():
        attrs = list(set([col.replace('_min', '').replace('_max', '') for col in v.index if '_min' in col or '_max' in col]))
        clauses = {attr: [v[f'{attr}_min'], v[f'{attr}_max']] for attr in attrs}
        predicate = Predicate(clauses=clauses)
        predicate.fit(test_data)
        for conformance in [1, .75, .5]:
            labels = get_labels(predicate, conformance)
            r = get_metrics(labels, predicate.mask, metrics)
            r['conformance'] = conformance
            r['vistype'] = v['vistype']
            res_list.append(r)
    test_res = pd.concat(res_list, axis=1).reset_index(drop=True).T
    test_res.columns = ['precision', 'recall', 'f1', 'conformance', 'vistype']
    return test_res[['vistype', 'conformance', 'precision', 'recall', 'f1']]

def get_final_res(args, all_data_train_test, all_predicates, all_params, all_metrics):
    all_res = {}
    for i,j in args.iterrows():
        args_keys = (j['data'], j['predicates'], j['params'], j['metrics'])
        print(i, args_keys)
        train_data, test_data = all_data_train_test[args_keys[0]]
        metrics = all_metrics[args_keys[3]]
        train_res = get_train_res(
            train_data, all_predicates[args_keys[1]],
            all_params[args_keys[2]], metrics
        )
        test_res = get_test_res(train_res, test_data, metrics)
        all_res[args_keys] = {'train': train_res, 'test': test_res}
    return all_res