import numpy as np
import pandas as pd
from vigor.predicate import Predicate

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
    numeric_ranges = {col: (data.kwargs[col][1]-data.kwargs[col][0])*numeric_p for col in numeric}
    
    left = {k: np.random.uniform(data.kwargs[k][0], data.kwargs[k][1]-v) for k,v in numeric_ranges.items()}
    intervals = {k: [v, v+numeric_ranges[k]] for k,v in left.items()}
    bin_left = {k: np.random.randint(0, 1) for k in binary}
    bin_intervals = {k: [v, v] for k,v in bin_left.items()}
    clauses = {k: v for k,v in intervals.items()}
    for k,v in bin_intervals.items():
        clauses[k] = v   
    return Predicate(clauses=clauses)

def test_predicate_proportions(all_predicates, all_data):
    res = {}    
    for predicates_name,predicates in all_predicates.items():
        for vistype,predicate in predicates.items():
            proportions = {}
            for data_name,df in all_data.items():
                predicate.fit(df)
                proportions[data_name] = predicate.mask.mean()
            res[(vistype, predicates_name)] = pd.Series(proportions)
    res_df = pd.DataFrame(res).T.reset_index()
    res_df.columns = ['vistype', 'predicates'] + res_df.columns[-2:].tolist()
    return res_df