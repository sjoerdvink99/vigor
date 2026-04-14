import argparse
import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from src import Predicate
from src.guidelines import predicates as custom_predicates, nobre_predicates
from evaluation.pipeline import generate_graphs, label_graphs, learn_predicates
from evaluation.baselines import fit_baselines, predicate_predict
from evaluation.datasets import one_hot_graph_type
from evaluation.metrics import (
    compute_metrics, compute_model_recovery, significance_tests
)
from evaluation.figures import (
    fig_recovery_bar, fig_recovery_heatmap, fig_recovery_scatter,
    fig_model_recovery, fig_learning_curves, fig_noise_robustness,
    fig_ablation, fig_transfer,
)


def _seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def _prep(df):
    return df.replace([np.inf, -np.inf], np.nan).fillna(0)


def _nobre_initial():
    prior_clauses = defaultdict(dict)
    for vistype, attr, minval, maxval in nobre_predicates:
        prior_clauses[vistype.name][attr] = [minval, maxval]
    return {k: Predicate(clauses=v) for k, v in prior_clauses.items()}


def _custom_initial():
    prior_clauses = defaultdict(dict)
    for vistype, attr, minval, maxval in custom_predicates:
        prior_clauses[vistype.name][attr] = [minval, maxval]
    return {k: Predicate(clauses=v) for k, v in prior_clauses.items()}


def _nobre_dict():
    d = defaultdict(dict)
    for vistype, attr, minval, maxval in nobre_predicates:
        d[vistype.name][attr] = [minval, maxval]
    return dict(d)


def run_recovery(n_seeds=3, output_dir="results/recovery", n_graphs=500,
                 n_train=400, nodes_max=100):
    os.makedirs(output_dir, exist_ok=True)
    n_test = n_graphs - n_train
    rows = []
    initial_predicates = _nobre_initial()

    print(f"[Recovery] n_graphs={n_graphs}, n_train={n_train}, seeds={n_seeds}")

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        _seed(seed)

        df = _prep(one_hot_graph_type(generate_graphs(n_graphs, nodes_max=nodes_max)))
        labels = label_graphs(df, nobre_predicates, conformance=1.0, mode="probability")

        active = labels.value_counts()
        active = active[active >= 5].index.tolist()
        labels = labels[labels.isin(active)]
        df_active = df.loc[labels.index]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df_active, labels, test_size=n_test, random_state=seed, stratify=labels
            )
        except ValueError:
            X_train = df_active.iloc[:n_train]
            X_test = df_active.iloc[n_train:]
            y_train = labels.iloc[:n_train]
            y_test = labels.iloc[n_train:]

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        for condition, init_preds in [("cold", None), ("prior", initial_predicates)]:
            learned = learn_predicates(X_train, y_train, top_k=12, initial_predicates=init_preds)
            metrics = compute_metrics(nobre_predicates, learned, X_train, X_test, y_train, y_test)
            for vis, m in metrics.items():
                for stat, scores in m["exact"].items():
                    rows.append({
                        "seed": seed, "vis": vis, "stat": stat, "condition": condition,
                        **scores,
                        "describe_f1": m["describe"]["f1"],
                        "generalize_f1": m["generalize"]["f1"],
                    })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "raw.csv"), index=False)

    summary = (
        results_df.groupby(["condition", "vis"])[["iou", "giou", "center_bias", "width_ratio"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.to_csv(os.path.join(output_dir, "summary.csv"))

    if "condition" in results_df.columns and results_df["condition"].nunique() > 1:
        stats = significance_tests(results_df, "iou", "condition", "cold", ["prior"])
        stats.to_csv(os.path.join(output_dir, "stats.csv"), index=False)

    print("\n[Recovery] Mean IoU by vis type:")
    for cond in sorted(results_df["condition"].unique()):
        s = results_df[results_df["condition"] == cond].groupby("vis")["iou"].mean()
        print(f"  [{cond}]")
        for vis, val in s.sort_values(ascending=False).items():
            print(f"    {vis:<30s} {val:.3f}")

    fig_recovery_bar(results_df, os.path.join(output_dir, "fig_recovery_bar.pdf"))
    fig_recovery_heatmap(results_df, os.path.join(output_dir, "fig_recovery_heatmap.pdf"))
    fig_recovery_scatter(results_df, os.path.join(output_dir, "fig_recovery_scatter.pdf"))

    print(f"[Recovery] Saved → {output_dir}/")


def run_model_recovery(n_seeds=3, output_dir="results/model_recovery", n_graphs=500,
                       nodes_max=100):
    os.makedirs(output_dir, exist_ok=True)
    nobre_d = _nobre_dict()
    initial_predicates = _nobre_initial()
    all_matrices = {"cold": [], "prior": []}

    print(f"[ModelRecovery] n_graphs={n_graphs}, seeds={n_seeds}")

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        _seed(seed)

        df = _prep(one_hot_graph_type(generate_graphs(n_graphs, nodes_max=nodes_max)))
        labels = label_graphs(df, nobre_predicates, conformance=1.0, mode="probability")

        active = labels.value_counts()
        active = active[active >= 5].index.tolist()
        labels = labels[labels.isin(active)]
        df_active = df.loc[labels.index].reset_index(drop=True)
        labels = labels.reset_index(drop=True)

        for condition, init_preds in [("cold", None), ("prior", initial_predicates)]:
            learned = learn_predicates(df_active, labels, initial_predicates=init_preds)
            if not learned:
                continue
            matrix = compute_model_recovery(learned, nobre_d)
            all_matrices[condition].append(matrix)

    mean_matrices = {}
    for condition, matrices in all_matrices.items():
        if not matrices:
            continue
        vis_types = matrices[0].index.tolist()
        nobre_types = matrices[0].columns.tolist()
        stacked = np.stack([m.values for m in matrices], axis=0)
        mean_matrices[condition] = pd.DataFrame(
            stacked.mean(axis=0), index=vis_types, columns=nobre_types
        )
        mean_matrices[condition].to_csv(
            os.path.join(output_dir, f"confusion_{condition}.csv")
        )

    if not mean_matrices:
        print("[ModelRecovery] No results generated.")
        return

    fig_model_recovery(mean_matrices, os.path.join(output_dir, "fig_model_recovery.pdf"))
    print(f"[ModelRecovery] Saved → {output_dir}/")


def run_sample_efficiency(n_seeds=3, output_dir="results/sample", n_graphs=600,
                          nodes_max=100, test_size=100):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    max_train = n_graphs - test_size
    train_sizes = [s for s in [20, 50, 100, 200, 400] if s <= max_train]
    noise_levels = [0.0, 0.25]

    print(f"[Sample] train_sizes={train_sizes}, noise={noise_levels}, seeds={n_seeds}")

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        _seed(seed)

        df = _prep(one_hot_graph_type(generate_graphs(n_graphs, nodes_max=nodes_max)))

        X_pool = df.iloc[:-test_size].reset_index(drop=True)
        X_test = df.iloc[-test_size:].reset_index(drop=True)

        for noise in noise_levels:
            conformance = 1.0 - noise
            all_labels = label_graphs(df, custom_predicates, conformance=conformance,
                                      mode="probability")
            y_test = all_labels.iloc[-test_size:].reset_index(drop=True)
            y_pool = all_labels.iloc[:-test_size].reset_index(drop=True)

            for n_train in train_sizes:
                try:
                    idx_train, _ = train_test_split(
                        np.arange(len(X_pool)), train_size=n_train,
                        random_state=seed, stratify=y_pool
                    )
                except ValueError:
                    idx_train = np.arange(min(n_train, len(X_pool)))

                X_train = X_pool.iloc[idx_train]
                y_train = y_pool.iloc[idx_train]

                non_const = X_train.columns[X_train.nunique() > 1]
                X_tr = X_train[non_const]
                X_te = X_test[non_const]
                min_v, max_v = X_tr.min(), X_tr.max()
                X_tr_norm = (X_tr - min_v) / (max_v - min_v + 1e-8)
                X_te_norm = (X_te - min_v) / (max_v - min_v + 1e-8)

                learned = learn_predicates(X_tr, y_train)
                pred_labels = predicate_predict(learned, X_te)
                f1_vigor = f1_score(y_test, pred_labels, average="macro", zero_division=0)
                rows.append({"seed": seed, "noise": noise, "n_train": n_train,
                             "method": "VigorPredicates", "f1": f1_vigor})

                baseline_preds = fit_baselines(X_tr_norm, y_train, X_te_norm, seed=seed)
                for method, preds in baseline_preds.items():
                    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
                    rows.append({"seed": seed, "noise": noise, "n_train": n_train,
                                 "method": method, "f1": f1})

                manual_preds = label_graphs(X_test, custom_predicates, conformance=1.0,
                                            mode="probability")
                f1_manual = f1_score(y_test, manual_preds, average="macro", zero_division=0)
                rows.append({"seed": seed, "noise": noise, "n_train": n_train,
                             "method": "ManualPredicates", "f1": f1_manual})

                print(f"    noise={noise:.2f} n_train={n_train:4d} | Vigor={f1_vigor:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "raw.csv"), index=False)
    fig_learning_curves(results_df, os.path.join(output_dir, "fig_learning_curves.pdf"))
    print(f"[Sample] Saved → {output_dir}/")


def run_noise(n_seeds=3, output_dir="results/noise", n_graphs=600, nodes_max=100,
              test_size=100, n_train=400):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    noise_levels = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]

    print(f"[Noise] n_train={n_train}, noise_levels={noise_levels}, seeds={n_seeds}")

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        _seed(seed)

        df = _prep(one_hot_graph_type(generate_graphs(n_graphs, nodes_max=nodes_max)))
        X_test = df.iloc[-test_size:].reset_index(drop=True)
        X_pool = df.iloc[:-test_size].reset_index(drop=True)

        try:
            all_labels_clean = label_graphs(df, custom_predicates, conformance=1.0,
                                            mode="probability")
            idx_train, _ = train_test_split(
                np.arange(len(X_pool)), train_size=n_train,
                random_state=seed,
                stratify=all_labels_clean.iloc[:-test_size]
            )
        except ValueError:
            idx_train = np.arange(min(n_train, len(X_pool)))

        X_train = X_pool.iloc[idx_train].reset_index(drop=True)

        non_const = X_train.columns[X_train.nunique() > 1]
        X_tr_raw = X_train[non_const]
        X_te_raw = X_test[non_const]
        min_v, max_v = X_tr_raw.min(), X_tr_raw.max()
        X_tr_norm = (X_tr_raw - min_v) / (max_v - min_v + 1e-8)
        X_te_norm = (X_te_raw - min_v) / (max_v - min_v + 1e-8)

        y_test = all_labels_clean.iloc[-test_size:].reset_index(drop=True)

        for noise_type in ["symmetric", "asymmetric"]:
            asym = noise_type == "asymmetric"
            for noise in noise_levels:
                conformance = 1.0 - noise
                noisy_labels = label_graphs(df, custom_predicates, conformance=conformance,
                                            mode="probability", asymmetric_noise=asym)
                y_train_noise = noisy_labels.iloc[:-test_size].iloc[idx_train].reset_index(drop=True)

                learned = learn_predicates(X_tr_raw, y_train_noise)
                pred_labels = predicate_predict(learned, X_te_raw)
                f1_vigor = f1_score(y_test, pred_labels, average="macro", zero_division=0)
                rows.append({"seed": seed, "noise": noise, "noise_type": noise_type,
                             "n_train": n_train, "method": "VigorPredicates", "f1": f1_vigor})

                baseline_preds = fit_baselines(X_tr_norm, y_train_noise, X_te_norm, seed=seed)
                for method, preds in baseline_preds.items():
                    f1 = f1_score(y_test, preds, average="macro", zero_division=0)
                    rows.append({"seed": seed, "noise": noise, "noise_type": noise_type,
                                 "n_train": n_train, "method": method, "f1": f1})

                manual_preds = label_graphs(X_test, custom_predicates, conformance=1.0,
                                            mode="probability")
                f1_manual = f1_score(y_test, manual_preds, average="macro", zero_division=0)
                rows.append({"seed": seed, "noise": noise, "noise_type": noise_type,
                             "n_train": n_train, "method": "ManualPredicates", "f1": f1_manual})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "raw.csv"), index=False)
    fig_noise_robustness(results_df, os.path.join(output_dir, "fig_noise_robustness.pdf"))
    print(f"[Noise] Saved → {output_dir}/")


def run_ablation(n_seeds=3, output_dir="results/ablation", n_graphs=500, nodes_max=100,
                 n_train=400):
    os.makedirs(output_dir, exist_ok=True)

    ablation_variants = {
        "Full system":         {},
        "b = 1 (linear)":     {"exponent": 1},
        "b = 2 (quadratic)":  {"exponent": 2},
        "b = 5 (steep)":      {"exponent": 5},
        "No smoothness":      {"smoothness_scale": 0},
        "No weight decay":    {"weight_decay_a": 0},
        "No momentum":        {"momentum": 0, "nesterov": False},
        "No feature select.": {"_top_k": None, "_n_restarts": 2},
        "1 restart":          {"_n_restarts": 1},
        "No rand. restart":   {"random_init": False},
    }

    rows = []
    print(f"[Ablation] variants={list(ablation_variants)}, seeds={n_seeds}")

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        _seed(seed)

        df = _prep(one_hot_graph_type(generate_graphs(n_graphs, nodes_max=nodes_max)))
        labels = label_graphs(df, custom_predicates, conformance=1.0, mode="probability")

        active = labels.value_counts()
        active = active[active >= 5].index.tolist()
        labels = labels[labels.isin(active)]
        df_active = df.loc[labels.index]

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                df_active, labels, test_size=n_graphs - n_train,
                random_state=seed, stratify=labels
            )
        except ValueError:
            X_train = df_active.iloc[:n_train]
            X_test = df_active.iloc[n_train:]
            y_train = labels.iloc[:n_train]
            y_test = labels.iloc[n_train:]

        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test = y_test.reset_index(drop=True)

        for variant_name, spec in ablation_variants.items():
            top_k = spec.get("_top_k", 12)
            n_restarts = spec.get("_n_restarts", 2)
            model_kwargs = {k: v for k, v in spec.items() if not k.startswith("_")}

            learned = learn_predicates(X_train, y_train, top_k=top_k,
                                       n_restarts=n_restarts, model_kwargs=model_kwargs)
            metrics = compute_metrics(custom_predicates, learned, X_train, X_test, y_train, y_test)

            iou_vals, f1_vals = [], []
            for vis, m in metrics.items():
                for stat, s in m["exact"].items():
                    iou_vals.append(s["iou"])
                f1_vals.append(m["generalize"]["f1"])

            rows.append({
                "seed": seed,
                "variant": variant_name,
                "iou": float(np.mean(iou_vals)) if iou_vals else 0.0,
                "f1": float(np.mean(f1_vals)) if f1_vals else 0.0,
            })
            print(f"    {variant_name:<25s} IoU={rows[-1]['iou']:.3f}  F1={rows[-1]['f1']:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "raw.csv"), index=False)

    if results_df["variant"].nunique() > 1:
        alts = [v for v in results_df["variant"].unique() if v != "Full system"]
        stats = significance_tests(results_df, "iou", "variant", "Full system", alts)
        stats.to_csv(os.path.join(output_dir, "stats.csv"), index=False)

    fig_ablation(results_df, os.path.join(output_dir, "fig_ablation.pdf"))
    print(f"[Ablation] Saved → {output_dir}/")


def run_transfer(n_seeds=3, output_dir="results/transfer", n_train=400, n_test=200,
                 nodes_max=100):
    os.makedirs(output_dir, exist_ok=True)
    rows = []

    print(f"[Transfer] n_train={n_train}, n_test={n_test}, seeds={n_seeds}")
    print("  Train families: GNP + BA  |  Test families: WS + SBM + tree + cycle")

    for seed in range(n_seeds):
        print(f"  Seed {seed + 1}/{n_seeds}")
        _seed(seed)

        df_train = _prep(one_hot_graph_type(
            generate_graphs(n_train, nodes_max=nodes_max, families={'gnp', 'ba'})
        ))
        df_test = _prep(one_hot_graph_type(
            generate_graphs(n_test, nodes_max=nodes_max, families={'ws', 'sbm', 'tree', 'cycle'})
        ))

        y_train = label_graphs(df_train, custom_predicates, conformance=1.0, mode="probability")
        y_test = label_graphs(df_test, custom_predicates, conformance=1.0, mode="probability")

        non_const = df_train.columns[df_train.nunique() > 1]
        X_tr = df_train[non_const]
        X_te = df_test.reindex(columns=df_train.columns, fill_value=0)[non_const]
        min_v, max_v = X_tr.min(), X_tr.max()
        X_tr_norm = (X_tr - min_v) / (max_v - min_v + 1e-8)
        X_te_norm = (X_te - min_v) / (max_v - min_v + 1e-8)

        learned = learn_predicates(X_tr, y_train)
        pred_labels = predicate_predict(learned, X_te)
        f1_vigor = f1_score(y_test, pred_labels, average="macro", zero_division=0)
        rows.append({"seed": seed, "method": "VigorPredicates", "f1": f1_vigor})

        baseline_preds = fit_baselines(X_tr_norm, y_train, X_te_norm, seed=seed)
        for method, preds in baseline_preds.items():
            f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            rows.append({"seed": seed, "method": method, "f1": f1})

        manual_preds = label_graphs(df_test, custom_predicates, conformance=1.0, mode="probability")
        f1_manual = f1_score(y_test, manual_preds, average="macro", zero_division=0)
        rows.append({"seed": seed, "method": "ManualPredicates", "f1": f1_manual})

        print(f"  Vigor F1={f1_vigor:.3f}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(output_dir, "raw.csv"), index=False)
    fig_transfer(results_df, os.path.join(output_dir, "fig_transfer.pdf"))
    print(f"[Transfer] Saved → {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Vigor evaluation runner")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("recovery")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-graphs", type=int, default=500)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--nodes-max", type=int, default=100)
    p.add_argument("--output-dir", default="results/recovery")

    p = sub.add_parser("model-recovery")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-graphs", type=int, default=500)
    p.add_argument("--nodes-max", type=int, default=100)
    p.add_argument("--output-dir", default="results/model_recovery")

    p = sub.add_parser("sample")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-graphs", type=int, default=600)
    p.add_argument("--nodes-max", type=int, default=100)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--output-dir", default="results/sample")

    p = sub.add_parser("noise")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-graphs", type=int, default=600)
    p.add_argument("--nodes-max", type=int, default=100)
    p.add_argument("--test-size", type=int, default=100)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--output-dir", default="results/noise")

    p = sub.add_parser("ablation")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-graphs", type=int, default=500)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--nodes-max", type=int, default=100)
    p.add_argument("--output-dir", default="results/ablation")

    p = sub.add_parser("transfer")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--nodes-max", type=int, default=100)
    p.add_argument("--output-dir", default="results/transfer")

    p = sub.add_parser("all")
    p.add_argument("--output-dir", default="results")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n-graphs", type=int, default=500)
    p.add_argument("--nodes-max", type=int, default=100)

    args = parser.parse_args()

    if args.command == "recovery":
        run_recovery(n_seeds=args.seeds, output_dir=args.output_dir,
                     n_graphs=args.n_graphs, n_train=args.n_train, nodes_max=args.nodes_max)
    elif args.command == "model-recovery":
        run_model_recovery(n_seeds=args.seeds, output_dir=args.output_dir,
                           n_graphs=args.n_graphs, nodes_max=args.nodes_max)
    elif args.command == "sample":
        run_sample_efficiency(n_seeds=args.seeds, output_dir=args.output_dir,
                              n_graphs=args.n_graphs, nodes_max=args.nodes_max,
                              test_size=args.test_size)
    elif args.command == "noise":
        run_noise(n_seeds=args.seeds, output_dir=args.output_dir,
                  n_graphs=args.n_graphs, nodes_max=args.nodes_max,
                  test_size=args.test_size, n_train=args.n_train)
    elif args.command == "ablation":
        run_ablation(n_seeds=args.seeds, output_dir=args.output_dir,
                     n_graphs=args.n_graphs, nodes_max=args.nodes_max,
                     n_train=args.n_train)
    elif args.command == "transfer":
        run_transfer(n_seeds=args.seeds, output_dir=args.output_dir,
                     n_train=args.n_train, n_test=args.n_test, nodes_max=args.nodes_max)
    elif args.command == "all":
        od = args.output_dir
        run_recovery(n_seeds=args.seeds, n_graphs=args.n_graphs,
                     nodes_max=args.nodes_max, output_dir=os.path.join(od, "recovery"))
        run_model_recovery(n_seeds=args.seeds, n_graphs=args.n_graphs,
                           nodes_max=args.nodes_max,
                           output_dir=os.path.join(od, "model_recovery"))
        run_sample_efficiency(n_seeds=args.seeds, n_graphs=args.n_graphs + 100,
                              nodes_max=args.nodes_max,
                              output_dir=os.path.join(od, "sample"))
        run_noise(n_seeds=args.seeds, n_graphs=args.n_graphs + 100,
                  nodes_max=args.nodes_max, output_dir=os.path.join(od, "noise"))
        run_ablation(n_seeds=args.seeds, n_graphs=args.n_graphs,
                     nodes_max=args.nodes_max, output_dir=os.path.join(od, "ablation"))
        run_transfer(n_seeds=args.seeds, nodes_max=args.nodes_max,
                     output_dir=os.path.join(od, "transfer"))


if __name__ == "__main__":
    main()
