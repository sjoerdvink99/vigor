import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Times"],
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.framealpha": 0.0,
    "legend.edgecolor": "none",
    "legend.borderpad": 0.3,
    "legend.labelspacing": 0.25,
    "legend.handlelength": 1.8,
    "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_W1 = 3.5
_W2 = 7.2

_IOUcmap = LinearSegmentedColormap.from_list("iou", ["#F7F7F7", "#4DAF4A"])
_CONFcmap = LinearSegmentedColormap.from_list("conf", ["#F7F7F7", "#2166AC"])

_METHODS = {
    "VigorPredicates":    {"label": "Vigor (ours)",       "color": "#C0392B", "marker": "o",  "ls": "-",  "lw": 1.8, "ms": 4.0, "zorder": 5},
    "ManualPredicates":   {"label": "Manual predicates",  "color": "#8E44AD", "marker": None, "ls": ":",  "lw": 1.4, "ms": 3.2, "zorder": 4},
    "DecisionTree":       {"label": "Decision Tree",      "color": "#2980B9", "marker": "s",  "ls": "--", "lw": 1.1, "ms": 3.2, "zorder": 3},
    "RandomForest":       {"label": "Random Forest",      "color": "#27AE60", "marker": "^",  "ls": "--", "lw": 1.1, "ms": 3.2, "zorder": 3},
    "LogisticRegression": {"label": "Logistic Reg.",      "color": "#E67E22", "marker": "D",  "ls": "-.", "lw": 1.1, "ms": 3.2, "zorder": 3},
    "MostFrequent":       {"label": "Most Frequent",      "color": "#AAAAAA", "marker": "x",  "ls": ":",  "lw": 0.9, "ms": 3.2, "zorder": 2},
}

_CONDITIONS = {
    "cold":  {"label": "Cold start",  "color": "#7F8C8D", "ecolor": "#555555"},
    "prior": {"label": "With priors", "color": "#C0392B", "ecolor": "#7B1010"},
}

_NOISE_TYPES = {
    "symmetric":  {"label": "Symmetric",  "color": "#2980B9", "ls": "--"},
    "asymmetric": {"label": "Asymmetric", "color": "#E67E22", "ls": "-."},
}

_ABLATION_FULL_COLOR = "#C0392B"
_ABLATION_BASE_COLOR = "#95A5A6"

_BOUND_LOWER = {"color": "#2980B9", "marker": "o", "s": 12, "alpha": 0.7}
_BOUND_UPPER = {"color": "#E74C3C", "marker": "^", "s": 12, "alpha": 0.7}


def _style(method):
    return _METHODS.get(method, {"label": method, "color": "#555555", "marker": ".",
                                  "ls": "-", "lw": 1.0, "ms": 3, "zorder": 1})


def _clean_ax(ax, grid_axis="y"):
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(axis="both", direction="out", length=3, width=0.7, pad=2)
    ax.set_axisbelow(True)
    if grid_axis == "y":
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.85", zorder=0)
    elif grid_axis == "x":
        ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="0.85", zorder=0)
    elif grid_axis == "both":
        ax.yaxis.grid(True, linestyle=":", linewidth=0.4, color="0.85", zorder=0)
        ax.xaxis.grid(True, linestyle=":", linewidth=0.4, color="0.85", zorder=0)


def _add_legend_below(fig, ax, ncol=3):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center",
               bbox_to_anchor=(0.5, 0.0), ncol=ncol,
               borderaxespad=0, frameon=False)


def _ci(series_grouped):
    return 1.96 * series_grouped.std().fillna(0) / np.sqrt(series_grouped.count().clip(lower=1))


def _line_plot(ax, df, x_col, methods_order):
    for method in methods_order:
        m = df[df["method"] == method]
        if m.empty:
            continue
        s = _style(method)
        grp = m.groupby(x_col)["f1"]
        means = grp.mean()
        ci = _ci(grp)
        ax.plot(means.index, means.values,
                label=s["label"], color=s["color"],
                marker=s["marker"], linestyle=s["ls"],
                linewidth=s["lw"], markersize=s["ms"],
                zorder=s["zorder"], clip_on=False)
        ax.fill_between(means.index,
                        (means - ci).clip(lower=0),
                        (means + ci).clip(upper=1),
                        alpha=0.12, color=s["color"], linewidth=0, zorder=s["zorder"] - 1)


def _sig_label(p):
    if np.isnan(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def fig_recovery_bar(df, path, stats_df=None):
    conditions = [c for c in ("cold", "prior") if c in df["condition"].unique()]
    n_cond = len(conditions)

    vis_order = (
        df[df["condition"] == conditions[0]].groupby("vis")["iou"].mean()
        .sort_values(ascending=True).index.tolist()
    )
    n_vis = len(vis_order)
    vis_labels = [v.replace("_", " ").title() for v in vis_order]

    bar_h = 0.28 if n_cond > 1 else 0.45
    fig_h = max(2.6, n_vis * (bar_h * n_cond + 0.28) + 0.9)
    fig, axes = plt.subplots(1, 2, figsize=(_W2, fig_h),
                             gridspec_kw={"wspace": 0.06, "width_ratios": [1, 1]})

    for ax_idx, (ax, metric) in enumerate(zip(axes, ["iou", "giou"])):
        _clean_ax(ax, grid_axis="x")
        for ci_idx, cond in enumerate(conditions):
            sub = df[df["condition"] == cond]
            grp = sub.groupby("vis")[metric]
            means = grp.mean().reindex(vis_order)
            err = _ci(grp).reindex(vis_order)
            offset = (ci_idx - (n_cond - 1) / 2.0) * (bar_h + 0.06)
            y = np.arange(n_vis) + offset
            s = _CONDITIONS[cond]
            ax.barh(y, means.values, xerr=err.values,
                    color=s["color"], height=bar_h,
                    label=s["label"] if ax_idx == 0 else "_nolegend_",
                    error_kw=dict(elinewidth=0.7, capsize=2.0, ecolor=s["ecolor"]),
                    zorder=3)

        ax.set_yticks(np.arange(n_vis))
        if ax_idx == 0:
            ax.set_yticklabels(vis_labels, fontsize=7)
            ax.set_xlabel("Avg. IoU")
        else:
            ax.set_yticklabels([])
            ax.set_xlabel("Avg. GIoU")
        ax.set_xlim(-0.25, 1.05)
        ax.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.axvline(0, color="0.5", linewidth=0.6, linestyle="--", zorder=2)
        ax.set_title(["Intersection over Union", "Generalized IoU"][ax_idx], fontsize=7.5, pad=4)

    axes[0].legend(loc="lower right", fontsize=6.5, handlelength=1.0)
    fig.savefig(path)
    plt.close(fig)


def fig_recovery_heatmap(df, path):
    conditions = [c for c in ("cold", "prior") if c in df["condition"].unique()]
    n_cond = len(conditions)

    vis_order = (
        df[df["condition"] == conditions[0]].groupby("vis")["iou"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    stat_order = (
        df.groupby("stat")["iou"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    n_vis, n_stat = len(vis_order), len(stat_order)
    vis_labels = [v.replace("_", " ").title() for v in vis_order]
    stat_labels = [s.replace("_", " ") for s in stat_order]

    fig_h = max(2.0, n_vis * 0.42 + 1.2)
    fig, axes = plt.subplots(
        1, n_cond,
        figsize=(_W2 if n_cond > 1 else _W1 + 1.0, fig_h),
        sharey=True,
        gridspec_kw={"wspace": 0.04},
    )
    if n_cond == 1:
        axes = [axes]

    im = None
    for idx, (ax, cond) in enumerate(zip(axes, conditions)):
        sub = df[df["condition"] == cond]
        pivot = (
            sub.groupby(["vis", "stat"])["iou"].mean()
            .unstack("stat")
            .reindex(index=vis_order, columns=stat_order)
        )

        im = ax.imshow(pivot.values, aspect="auto", cmap=_IOUcmap, vmin=0, vmax=1,
                       interpolation="nearest")

        ax.set_xticks(np.arange(n_stat))
        ax.set_xticklabels(stat_labels, rotation=40, ha="right", fontsize=6.0)
        ax.set_title(_CONDITIONS[cond]["label"], fontsize=7.5, pad=5)

        if idx == 0:
            ax.set_yticks(np.arange(n_vis))
            ax.set_yticklabels(vis_labels, fontsize=7)

        for i in range(n_vis):
            for j in range(n_stat):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    txt_color = "white" if val > 0.65 else "#333333"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                            fontsize=5.0, color=txt_color, fontweight="normal")

        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(axis="both", length=0, pad=3)
        ax.set_xticks(np.arange(n_stat) - 0.5, minor=True)
        ax.set_yticks(np.arange(n_vis) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linewidth=1.2)

    cbar = fig.colorbar(im, ax=axes, shrink=0.80, pad=0.015, aspect=30)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.set_label("IoU", fontsize=7)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])

    fig.savefig(path)
    plt.close(fig)


def fig_recovery_scatter(df, path):
    key_stats = [s for s in ["n_nodes", "node_types", "node_attributes",
                              "edge_types", "edge_attributes", "density",
                              "clustering_coefficient", "modularity"]
                 if s in df["stat"].unique()]
    if not key_stats:
        return

    n = len(key_stats)
    ncols = 4
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(_W2, nrows * 1.7 + 0.4),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.4})
    axes_flat = np.array(axes).flatten()

    for ax_i, stat in enumerate(key_stats):
        ax = axes_flat[ax_i]
        sub = df[df["stat"] == stat]
        if sub.empty:
            ax.set_visible(False)
            continue

        _clean_ax(ax, grid_axis="both")

        all_vals = np.concatenate([sub["lo_true"], sub["hi_true"],
                                   sub["lo_learned"], sub["hi_learned"]])
        all_vals = all_vals[np.isfinite(all_vals)]
        if len(all_vals) == 0:
            ax.set_visible(False)
            continue
        vmin, vmax = all_vals.min(), all_vals.max()
        pad = (vmax - vmin) * 0.08 if vmax > vmin else 0.5
        lim = (vmin - pad, vmax + pad)

        ax.plot(lim, lim, color="0.75", linewidth=0.8, linestyle="--", zorder=1)

        ax.scatter(sub["lo_true"], sub["lo_learned"],
                   **{**_BOUND_LOWER, "label": "Lower bd.", "edgecolors": "none", "zorder": 3})
        ax.scatter(sub["hi_true"], sub["hi_learned"],
                   **{**_BOUND_UPPER, "label": "Upper bd.", "edgecolors": "none", "zorder": 3})

        lo_true_arr = sub["lo_true"].values
        lo_learned_arr = sub["lo_learned"].values
        hi_true_arr = sub["hi_true"].values
        hi_learned_arr = sub["hi_learned"].values

        combined_true = np.concatenate([lo_true_arr, hi_true_arr])
        combined_learned = np.concatenate([lo_learned_arr, hi_learned_arr])
        finite_mask = np.isfinite(combined_true) & np.isfinite(combined_learned)
        if finite_mask.sum() > 2:
            r = np.corrcoef(combined_true[finite_mask], combined_learned[finite_mask])[0, 1]
            ax.text(0.06, 0.92, f"r = {r:.2f}", transform=ax.transAxes,
                    fontsize=6.0, va="top", color="#333333")

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(stat.replace("_", " "), fontsize=6.5, pad=3)
        if ax_i % ncols == 0:
            ax.set_ylabel("Recovered", fontsize=6.5)
        if ax_i >= (nrows - 1) * ncols:
            ax.set_xlabel("True", fontsize=6.5)

    for ax_i in range(len(key_stats), len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=_BOUND_LOWER["color"],
               markersize=4, label="Lower bd."),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=_BOUND_UPPER["color"],
               markersize=4, label="Upper bd."),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0),
               ncol=2, frameon=False, fontsize=7)
    fig.subplots_adjust(bottom=0.10)
    fig.savefig(path)
    plt.close(fig)


def _draw_recovery_panel(ax, data, vis_types, nobre_types, title, show_ylabel):
    n_true, n_pred = len(vis_types), len(nobre_types)
    im = ax.imshow(data, aspect="auto", cmap=_CONFcmap, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(np.arange(n_pred))
    ax.set_xticklabels([t.replace("_", "\n").title() for t in nobre_types],
                       fontsize=5.5, ha="center")
    ax.set_yticks(np.arange(n_true))
    ax.set_yticklabels(
        [t.replace("_", " ").title() for t in vis_types] if show_ylabel else [""] * n_true,
        fontsize=6.5
    )
    ax.set_xlabel("Reference type (Nobre et al.)", fontsize=7)
    if show_ylabel:
        ax.set_ylabel("Learned from", fontsize=7)
    ax.set_title(title, fontsize=8, pad=6)
    for i in range(n_true):
        for j in range(n_pred):
            val = data[i, j]
            is_diag = vis_types[i] == nobre_types[j]
            txt_color = "white" if val > 0.5 else "#333333"
            label = f"{val:.2f}" if val > 0.01 else ""
            weight = "bold" if (val >= 0.5 or is_diag) else "normal"
            ax.text(j, i, label, ha="center", va="center",
                    fontsize=5.8, color=txt_color, fontweight=weight)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0, pad=4)
    ax.set_xticks(np.arange(n_pred) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_true) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=1.5)
    return im


def fig_model_recovery(matrices, path):
    if isinstance(matrices, dict):
        conditions = list(matrices.keys())
        dfs = list(matrices.values())
    else:
        conditions = ["Model Recovery"]
        dfs = [matrices]

    vis_types = dfs[0].index.tolist()
    nobre_types = dfs[0].columns.tolist()
    n_panels = len(conditions)

    _CONDITION_TITLES = {"cold": "Cold start", "prior": "With priors"}
    titles = [_CONDITION_TITLES.get(c, c.replace("_", " ").title()) for c in conditions]

    fig, axes = plt.subplots(1, n_panels,
                             figsize=(_W2 if n_panels > 1 else _W1 + 0.8,
                                      max(3.5, len(vis_types) * 0.55 + 1.2)),
                             gridspec_kw={"wspace": 0.05})
    if n_panels == 1:
        axes = [axes]

    im = None
    for idx, (ax, df, title) in enumerate(zip(axes, dfs, titles)):
        im = _draw_recovery_panel(ax, df.values.astype(float), vis_types, nobre_types,
                                  title, show_ylabel=(idx == 0))

    cbar = fig.colorbar(im, ax=axes[-1], shrink=0.7, pad=0.03, aspect=25)
    cbar.ax.tick_params(labelsize=6.5, length=2)
    cbar.set_label("Mean GIoU", fontsize=7)

    fig.savefig(path)
    plt.close(fig)


def fig_learning_curves(df, path):
    noise_levels = sorted(df["noise"].unique())
    show_noise = [n for n in [0.0, 0.25] if n in noise_levels]
    if not show_noise:
        show_noise = noise_levels[:2]
    n_panels = len(show_noise)

    conformance_labels = {0.0: "Informed (c = 1.0)", 0.25: "Semi-informed (c = 0.75)"}

    fig, axes = plt.subplots(1, n_panels, figsize=(_W2, 2.8), sharey=True,
                             gridspec_kw={"wspace": 0.06})
    if n_panels == 1:
        axes = [axes]

    methods_order = list(_METHODS.keys())

    for panel_idx, (ax, noise) in enumerate(zip(axes, show_noise)):
        sub = df[df["noise"] == noise]
        _clean_ax(ax)
        _line_plot(ax, sub, "n_train", methods_order)

        train_sizes = sorted(sub["n_train"].unique().tolist())
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(mticker.FixedLocator(train_sizes))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.xaxis.set_major_formatter(mticker.FixedFormatter([str(s) for s in train_sizes]))
        ax.set_xlim(train_sizes[0] * 0.7, train_sizes[-1] * 1.5)
        ax.set_ylim(0, 1.07)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Training set size")
        if panel_idx == 0:
            ax.set_ylabel("Macro F1")
        ax.set_title(conformance_labels.get(noise, f"Noise = {noise:.2f}"), fontsize=7.5, pad=4)

        vigor_df = sub[sub["method"] == "VigorPredicates"].groupby("n_train")["f1"]
        if not vigor_df.ngroups:
            continue
        vigor_means = vigor_df.mean()
        if len(vigor_means) > 0:
            max_f1 = vigor_means.max()
            threshold_90 = vigor_means[vigor_means >= 0.9 * max_f1]
            if not threshold_90.empty:
                n_thresh = threshold_90.index[0]
                ax.axvline(n_thresh, color="#C0392B", linewidth=0.8, linestyle=":",
                           alpha=0.7, zorder=1)

    _add_legend_below(fig, axes[0], ncol=min(len(methods_order), 3))
    fig.subplots_adjust(bottom=0.30)
    fig.savefig(path)
    plt.close(fig)


def fig_sample_efficiency(df, path):
    fig_learning_curves(df, path)


def fig_noise_robustness(df, path):
    has_noise_type = "noise_type" in df.columns
    n_train_val = 400 if 400 in df["n_train"].values else int(df["n_train"].max())
    sub = df[df["n_train"] == n_train_val]
    methods_order = list(_METHODS.keys())

    if has_noise_type:
        noise_types = [t for t in ["symmetric", "asymmetric"] if t in sub["noise_type"].values]
        n_panels = len(noise_types)
        fig, axes = plt.subplots(1, n_panels, figsize=(_W2 if n_panels > 1 else _W1, 2.6),
                                 sharey=True, gridspec_kw={"wspace": 0.06})
        if n_panels == 1:
            axes = [axes]

        for panel_idx, (ax, nt) in enumerate(zip(axes, noise_types)):
            sub_nt = sub[sub["noise_type"] == nt]
            _clean_ax(ax)
            _line_plot(ax, sub_nt, "noise", methods_order)

            noise_vals = sorted(sub_nt["noise"].unique())
            ax.set_xticks(noise_vals)
            ax.set_xticklabels([f"{n:.1f}" for n in noise_vals])
            ax.set_xlim(noise_vals[0] - 0.02, noise_vals[-1] + 0.02)
            ax.set_ylim(0, 1.07)
            ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xlabel("Noise level")
            if panel_idx == 0:
                ax.set_ylabel("Macro F1")
            nt_label = _NOISE_TYPES.get(nt, {}).get("label", nt.capitalize())
            ax.set_title(f"{nt_label} noise", fontsize=7.5, pad=4)

        _add_legend_below(fig, axes[0], ncol=min(len(methods_order), 3))
        fig.subplots_adjust(bottom=0.32)
    else:
        fig, ax = plt.subplots(figsize=(_W1, 2.6))
        _clean_ax(ax)
        _line_plot(ax, sub, "noise", methods_order)
        noise_vals = sorted(sub["noise"].unique())
        ax.set_xticks(noise_vals)
        ax.set_xticklabels([f"{n:.1f}" for n in noise_vals])
        ax.set_xlim(noise_vals[0] - 0.02, noise_vals[-1] + 0.02)
        ax.set_ylim(0, 1.07)
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_xlabel("Noise level")
        ax.set_ylabel("Macro F1")
        _add_legend_below(fig, ax, ncol=3)
        fig.subplots_adjust(bottom=0.30)

    fig.savefig(path)
    plt.close(fig)


def fig_ablation(df, path):
    variant_order = (
        df.groupby("variant")["iou"].mean()
        .sort_values(ascending=True).reset_index()
    )
    variants = variant_order["variant"].tolist()
    n_vars = len(variants)
    iou_means = variant_order["iou"].tolist()

    f1_means = df.groupby("variant")["f1"].mean().reindex(variants).tolist()
    iou_err = df.groupby("variant")["iou"].apply(
        lambda x: 1.96 * x.std() / np.sqrt(max(len(x), 1))
    ).reindex(variants).tolist()

    colors = [_ABLATION_FULL_COLOR if "Full" in v else _ABLATION_BASE_COLOR for v in variants]

    fig, ax = plt.subplots(figsize=(_W2, max(2.8, n_vars * 0.38 + 1.0)))
    _clean_ax(ax, grid_axis="x")

    y = np.arange(n_vars)
    bars = ax.barh(y, iou_means, xerr=iou_err, color=colors, height=0.52,
                   error_kw=dict(elinewidth=0.7, capsize=2.0, ecolor="0.4"),
                   zorder=3)

    ax2 = ax.twiny()
    ax2.scatter(f1_means, y, color=[c if c == _ABLATION_FULL_COLOR else "#3498DB" for c in colors],
                marker="D", s=16, zorder=5, clip_on=False)
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel("Generalize F1 (markers)", fontsize=7)
    ax2.tick_params(axis="x", labelsize=6.5, length=2.5, pad=2)
    for spine in ["top"]:
        ax2.spines[spine].set_visible(True)
        ax2.spines[spine].set_linewidth(0.7)

    ax.set_yticks(y)
    ax.set_yticklabels([v.replace("_", " ") for v in variants], fontsize=7)
    ax.set_xlabel("Mean IoU (bars)", fontsize=7)
    ax.set_xlim(0, 1.05)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    full_idx = next((i for i, v in enumerate(variants) if "Full" in v), None)
    if full_idx is not None:
        ax.axhline(full_idx, color=_ABLATION_FULL_COLOR, linewidth=0.6,
                   linestyle="--", alpha=0.4, zorder=1)

    handles = [
        matplotlib.patches.Patch(color=_ABLATION_FULL_COLOR, label="Full system"),
        matplotlib.patches.Patch(color=_ABLATION_BASE_COLOR, label="Ablated variant"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor="#3498DB",
               markersize=4, label="Gen. F1"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=6.5, handlelength=1.0)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_transfer(df, path):
    methods = df["method"].unique().tolist()
    if "domain" in df.columns:
        domains = df["domain"].unique().tolist()
        has_domains = True
    else:
        has_domains = False

    fig, axes = plt.subplots(1, 2 if has_domains else 1,
                             figsize=(_W2 if has_domains else _W1, 2.6),
                             gridspec_kw={"wspace": 0.08, "width_ratios": [1, 1.4]} if has_domains else {})

    if not has_domains:
        axes = [axes]

    ax = axes[0]
    _clean_ax(ax, grid_axis="x")

    grp = df.groupby("method")["f1"]
    means = grp.mean().reindex(methods)
    errs = _ci(grp).reindex(methods)
    colors = [_style(m)["color"] for m in methods]
    labels = [_style(m)["label"] for m in methods]

    y = np.arange(len(methods))
    bars = ax.barh(y, means.values, xerr=errs.values, color=colors, height=0.52,
                   error_kw=dict(elinewidth=0.7, capsize=2.0, ecolor="0.4"), zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Macro F1")
    ax.set_xlim(0, 1.12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.invert_yaxis()

    for bar, val in zip(bars, means.values):
        if np.isfinite(val):
            ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left", fontsize=6.0)

    vigor_idx = next((i for i, m in enumerate(methods) if m == "VigorPredicates"), None)
    if vigor_idx is not None:
        bars[vigor_idx].set_edgecolor("#7B1010")
        bars[vigor_idx].set_linewidth(0.9)

    if has_domains:
        ax2 = axes[1]
        _clean_ax(ax2, grid_axis="x")
        domain_f1 = df.groupby(["method", "domain"])["f1"].mean().unstack("domain").reindex(methods)
        im = ax2.imshow(domain_f1.values, aspect="auto", cmap=_CONFcmap, vmin=0, vmax=1,
                        interpolation="nearest")
        ax2.set_xticks(np.arange(len(domains)))
        ax2.set_xticklabels(domains, rotation=30, ha="right", fontsize=6.5)
        ax2.set_yticks(np.arange(len(methods)))
        ax2.set_yticklabels([_style(m)["label"] for m in methods], fontsize=6.5)
        ax2.set_title("F1 per domain", fontsize=7.5, pad=4)
        for i, m in enumerate(methods):
            for j, d in enumerate(domains):
                val = domain_f1.values[i, j]
                if np.isfinite(val):
                    txt_color = "white" if val > 0.6 else "#333333"
                    ax2.text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=5.5, color=txt_color)
        for spine in ax2.spines.values():
            spine.set_visible(False)
        ax2.tick_params(length=0, pad=3)
        ax2.set_xticks(np.arange(len(domains)) - 0.5, minor=True)
        ax2.set_yticks(np.arange(len(methods)) - 0.5, minor=True)
        ax2.grid(which="minor", color="white", linewidth=1.2)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def fig_domain_classification(df, path):
    fig_transfer(df, path)
