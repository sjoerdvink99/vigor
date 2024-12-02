import pandas as pd

# class Predicate:
#     def __init__(self, statistic: str, min: float, max: float) -> None:
#         self.min = min
#         self.max = max
#         self.mu = (min + max) / 2
#         self.a = 1 / ((max - min) / 2)
#         self.statistic = statistic

#     def relevance(self, value: float) -> float:
#         """Returns a score based on the evaluation of the statistic."""
#         if value < self.min or value > self.max:
#             return 0
#         return (value - self.min) / (self.max - self.min)
    
class Predicate:

    def __init__(self, clause_list=None, mask=None, mask_=None, clauses=None):
        self.clause_list = clause_list
        self.clauses = {clause['attribute']: clause['interval'] for clause in clause_list} if clauses is None else clauses
        self.mask_ = mask
        self.mask = mask_
        self.X = None
        self.attrs = tuple(set(self.clauses.keys()))
        
    def __repr__(self):
        return str(self.clauses)
        
    def fit(self, X):
        self.X = X
        self.mask_ = pd.DataFrame({k: (X[k]<=v[1]) & (X[k]>=v[0]) for k,v in self.clauses.items() if k in X.columns})
        self.mask = self.mask_.all(axis=1)
        
    def copy(self):
        predicate = Predicate(None, self.mask, self.mask_, self.clauses)
        if self.X is not None:
            predicate.fit(self.X)
        return predicate
    
    def label(self, X, col, dim=None):
        if dim is not None:
            not_dim_mask = self.mask_.drop(dim, axis=1).all(axis=1)
            return self.mask_.loc[not_dim_mask,dim].astype(int)
        else:
            return self.mask.astype(int)        
    
    def update_clause(self, dim, a, b):
        clauses = {k:v for k,v in self.clauses.items()}
        clauses[dim] = [a, b]
        predicate = Predicate(None, clauses=clauses)
        if self.X is not None:
            predicate.fit(self.X)
        return predicate
        
    def label_adjacent(self, X, col, dim):
        p1 = self.update_clause(dim, X[dim].min(), self.clauses[dim][0])
        p2 = self.update_clause(dim, self.clauses[dim][1], X[dim].max())
        return p1, p2, p1.label(X, col, dim), p2.label(X, col, dim)