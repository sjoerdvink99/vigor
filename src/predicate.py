import pandas as pd


class Predicate:
    def __init__(self, clause_list=None, clauses=None):
        self.clauses = (
            {clause["attribute"]: clause["interval"] for clause in clause_list}
            if clauses is None
            else clauses
        )
        self.mask = None
        self.mask_ = None
        self.X = None
        self.attrs = tuple(set(self.clauses.keys()))

    def __repr__(self):
        return str(self.clauses)

    def fit(self, X):
        self.X = X
        self.mask_ = pd.DataFrame(
            {k: (X[k] <= v[1]) & (X[k] >= v[0]) for k, v in self.clauses.items() if k in X.columns},
            index=X.index,
        )
        self.mask = self.mask_.all(axis=1)

    def copy(self):
        predicate = Predicate(clauses=dict(self.clauses))
        if self.X is not None:
            predicate.fit(self.X)
        return predicate

    def label(self, X, col, dim=None):
        if dim is not None:
            not_dim_mask = self.mask_.drop(dim, axis=1).all(axis=1)
            return self.mask_.loc[not_dim_mask, dim].astype(int)
        return self.mask.astype(int)

    def update_clause(self, dim, a, b):
        clauses = {k: v for k, v in self.clauses.items()}
        clauses[dim] = [a, b]
        predicate = Predicate(clauses=clauses)
        if self.X is not None:
            predicate.fit(self.X)
        return predicate
