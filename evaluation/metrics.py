from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def accuracy_metrics(vistype, predicate, graphs, labels):
    y_true = (labels == vistype).astype(int)
    predicate.fit(graphs)
    y_pred = predicate.mask.astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0, average="binary"
    )
    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "accuracy": float((y_true == y_pred).mean()),
    }


def _giou(lo_i, hi_i, lo_l, hi_l):
    intersection = max(0.0, min(hi_i, hi_l) - max(lo_i, lo_l))
    union_len = (hi_i - lo_i) + (hi_l - lo_l) - intersection
    iou = intersection / union_len if union_len > 0 else 0.0
    enclosing = max(hi_i, hi_l) - min(lo_i, lo_l)
    if enclosing <= 0:
        return iou
    return iou - (enclosing - union_len) / enclosing


def _center_bias(lo_i, hi_i, lo_l, hi_l):
    return ((lo_l + hi_l) / 2.0) - ((lo_i + hi_i) / 2.0)


def _width_ratio(lo_i, hi_i, lo_l, hi_l):
    w_i = hi_i - lo_i
    w_l = hi_l - lo_l
    if w_i <= 0:
        return 1.0
    return w_l / w_i


def compute_metrics(initial, learned, graphs, test_graphs, labels, test_labels):
    initial_dict = defaultdict(dict)
    for vis_type, stat_name, min_val, max_val in initial:
        initial_dict[vis_type.name][stat_name] = [min_val, max_val]
    initial_dict = dict(initial_dict)

    evaluation = {}
    for vis in learned:
        if vis not in initial_dict:
            continue
        initial_pred = initial_dict[vis]
        learned_pred = learned[vis]
        stats = initial_pred.keys() & learned_pred.clauses.keys()

        interval_scores = {}
        for stat in stats:
            lo_i, hi_i = initial_pred[stat]
            lo_l, hi_l = float(learned_pred.clauses[stat][0]), float(learned_pred.clauses[stat][1])
            intersection = max(0.0, min(hi_i, hi_l) - max(lo_i, lo_l))
            union_len = (hi_i - lo_i) + (hi_l - lo_l) - intersection
            iou = intersection / union_len if union_len > 0 else 0.0
            deviation = (abs(lo_i - lo_l) + abs(hi_i - hi_l)) / 2.0
            inclusion = 1 if (lo_l >= lo_i and hi_l <= hi_i) or (lo_i >= lo_l and hi_i <= hi_l) else 0
            interval_scores[stat] = {
                "iou": iou,
                "giou": _giou(lo_i, hi_i, lo_l, hi_l),
                "deviation": deviation,
                "inclusion": inclusion,
                "center_bias": _center_bias(lo_i, hi_i, lo_l, hi_l),
                "width_ratio": _width_ratio(lo_i, hi_i, lo_l, hi_l),
                "lo_true": lo_i,
                "hi_true": hi_i,
                "lo_learned": lo_l,
                "hi_learned": hi_l,
            }

        evaluation[vis] = {
            "exact": interval_scores,
            "describe": accuracy_metrics(vis, learned_pred, graphs, labels),
            "generalize": accuracy_metrics(vis, learned_pred, test_graphs, test_labels),
        }

    return evaluation


def compute_model_recovery(learned_predicates, nobre_dict):
    vis_types = list(learned_predicates.keys())
    nobre_types = list(nobre_dict.keys())
    matrix = np.zeros((len(vis_types), len(nobre_types)))

    for i, true_type in enumerate(vis_types):
        learned = learned_predicates[true_type]
        for j, nobre_type in enumerate(nobre_types):
            nobre_clauses = nobre_dict.get(nobre_type, {})
            common = set(learned.clauses.keys()) & set(nobre_clauses.keys())
            if not common:
                score = 0.0
            else:
                ious = []
                for attr in common:
                    lo_l, hi_l = learned.clauses[attr]
                    lo_n, hi_n = nobre_clauses[attr]
                    ious.append(_giou(lo_n, hi_n, float(lo_l), float(hi_l)))
                score = float(np.mean(ious))
            matrix[i, j] = max(0.0, score)

    return pd.DataFrame(matrix, index=vis_types, columns=nobre_types)


def significance_tests(df, metric_col, condition_col, reference_condition, alternative_conditions):
    from scipy.stats import wilcoxon, friedmanchisquare
    results = []
    ref = df[df[condition_col] == reference_condition][metric_col].values

    all_groups = [df[df[condition_col] == c][metric_col].values
                  for c in [reference_condition] + alternative_conditions]
    try:
        stat_f, p_f = friedmanchisquare(*all_groups)
    except Exception:
        stat_f, p_f = np.nan, np.nan

    results.append({
        "test": "friedman",
        "condition": "all",
        "statistic": stat_f,
        "p_value": p_f,
        "effect_size": np.nan,
    })

    n_comparisons = len(alternative_conditions)
    for alt in alternative_conditions:
        alt_vals = df[df[condition_col] == alt][metric_col].values
        min_len = min(len(ref), len(alt_vals))
        if min_len < 2:
            continue
        try:
            stat_w, p_w = wilcoxon(ref[:min_len], alt_vals[:min_len])
            z = stat_w / np.sqrt(min_len * (min_len + 1) * (2 * min_len + 1) / 6)
            effect_r = abs(z) / np.sqrt(min_len)
            p_corrected = min(p_w * n_comparisons, 1.0)
        except Exception:
            stat_w, p_corrected, effect_r = np.nan, np.nan, np.nan
        results.append({
            "test": "wilcoxon",
            "condition": alt,
            "statistic": stat_w,
            "p_value": p_corrected,
            "effect_size": effect_r,
        })

    return pd.DataFrame(results)
