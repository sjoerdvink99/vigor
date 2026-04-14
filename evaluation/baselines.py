import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier


def fit_baselines(X_train, y_train, X_test, seed=0):
    classifiers = {
        "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=seed),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=seed),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=seed),
        "MostFrequent": DummyClassifier(strategy="most_frequent", random_state=seed),
    }

    predictions = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        predictions[name] = clf.predict(X_test)

    return predictions


def predicate_predict(learned_predicates, X_test):
    scores = {}
    for label, pred in learned_predicates.items():
        if not pred.clauses:
            scores[label] = pd.Series(0.0, index=X_test.index)
            continue
        n_clauses = 0
        s = pd.Series(0.0, index=X_test.index)
        for attr, (lo, hi) in pred.clauses.items():
            if attr not in X_test.columns:
                continue
            r = (hi - lo) / 2.0
            if r <= 0:
                continue
            mu = (lo + hi) / 2.0
            s += ((X_test[attr] - mu).abs() / r).pow(3)
            n_clauses += 1
        scores[label] = 1.0 / (1.0 + s / max(n_clauses, 1))

    if not scores:
        first_label = next(iter(learned_predicates.keys()))
        return np.array([first_label] * len(X_test))

    return pd.DataFrame(scores).idxmax(axis=1).values
