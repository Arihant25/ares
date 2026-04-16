import json
import os
import statistics
from collections import defaultdict, Counter
from scipy.stats import spearmanr, wilcoxon

# ---------------------------------------------------------------------------
# Color scheme — consistent across all figures (derived from latency plot)
# ---------------------------------------------------------------------------
APPROACH_COLORS = {
    'Baseline':     '#4C72B0',
    'Task-focused': '#DD8452',
    'Fine-tuned':   '#55A868',
}
PRIMARY_COLOR = '#4C72B0'

RUN_META = {
    'baseline_grok':       ('Grok 4.1 Fast', 'Baseline',     'LLM'),
    'baseline_qwen':       ('Qwen-3 235B',   'Baseline',     'LLM'),
    'baseline_smollm3':    ('SmolLM3-3B',    'Baseline',     'SLM'),
    'baseline_lfm2':       ('LFM2-700M',     'Baseline',     'SLM'),
    'taskfocused_grok':    ('Grok 4.1 Fast', 'Task-focused', 'LLM'),
    'taskfocused_qwen':    ('Qwen-3 235B',   'Task-focused', 'LLM'),
    'taskfocused_smollm3': ('SmolLM3-3B',    'Task-focused', 'SLM'),
    'taskfocused_lfm2':    ('LFM2-700M',     'Task-focused', 'SLM'),
    'finetuned_smollm3':   ('SmolLM3-3B',    'Fine-tuned',   'SLM'),
    'finetuned_lfm2':      ('LFM2-700M',     'Fine-tuned',   'SLM'),
}

MODELS_ORDER   = ['Grok 4.1 Fast', 'Qwen-3 235B', 'SmolLM3-3B', 'LFM2-700M']
APPROACHES     = ['Baseline', 'Task-focused', 'Fine-tuned']


def load_data(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def cohen_kappa_score_quadratic(y1, y2, min_r=1, max_r=5):
    n = len(y1)
    if n == 0:
        return 0.0
    scale = max_r - min_r
    w = lambda i, j: 1.0 - ((i - j) / scale) ** 2
    po = sum(w(y1[idx], y2[idx]) for idx in range(n)) / n
    c_a, c_b = Counter(y1), Counter(y2)
    pe = sum(
        w(i, j) * (c_a.get(i, 0) / n) * (c_b.get(j, 0) / n)
        for i in range(min_r, max_r + 1)
        for j in range(min_r, max_r + 1)
    )
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def compute_eval_scores(eval_data):
    """
    Returns:
      item_avg: dict (run_id, item_index) -> mean composite score across evaluators
      run_means: dict  run_id -> mean composite score
    """
    item_scores = defaultdict(list)
    for entry in eval_data:
        composite = sum(entry['scores'].values()) / len(entry['scores'])
        item_scores[(entry['run_id'], entry['item_index'])].append(composite)
    item_avg = {k: sum(v) / len(v) for k, v in item_scores.items()}
    run_scores = defaultdict(list)
    for (run_id, _), score in item_avg.items():
        run_scores[run_id].append(score)
    run_means = {rid: statistics.mean(s) for rid, s in run_scores.items()}
    return item_avg, run_means


def compute_category_scores(item_avg, root_dir):
    """Join eval scores with category labels from output files."""
    cat_scores = defaultdict(list)
    for run_key in RUN_META:
        fpath = os.path.join(root_dir, 'outputs', f'{run_key}.json')
        if not os.path.exists(fpath):
            continue
        data = load_data(fpath)
        for i, d in enumerate(data):
            score = item_avg.get((run_key, i))
            if score is not None:
                cat_scores[d['Category']].append(score)
    return {cat: statistics.mean(scores) for cat, scores in cat_scores.items()}


# ---------------------------------------------------------------------------
# Plot helpers — shared style
# ---------------------------------------------------------------------------
def _apply_shared_style(ax, ylabel, title=None):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=11)
    if title:
        ax.set_title(title, fontsize=11)
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------------------
# Figure 1 — Results plot (composite scores by model × approach)
# ---------------------------------------------------------------------------
def plot_results(run_means, plots_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping results_plot.pdf")
        return

    width   = 0.25
    offsets = [-width, 0, width]
    x       = np.arange(len(MODELS_ORDER))

    fig, ax = plt.subplots(figsize=(9, 4))

    for i, approach in enumerate(APPROACHES):
        scores = []
        for model in MODELS_ORDER:
            # Find the run_id that matches this (model, approach)
            score = None
            for run_id, (m, a, _) in RUN_META.items():
                if m == model and a == approach:
                    score = run_means.get(run_id)
                    break
            scores.append(score if score is not None else 0.0)
        bars = ax.bar(
            x + offsets[i], scores, width,
            label=approach,
            color=APPROACH_COLORS[approach],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5,
        )
        # Annotate only non-zero bars
        for bar, s in zip(bars, scores):
            if s > 0.0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.03,
                    f'{s:.2f}',
                    ha='center', va='bottom', fontsize=7,
                )

    # Separator between LLMs and SLMs
    ax.axvline(x=1.5, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
    ax.text(0.5, 0.97, 'LLM', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9, color='gray', style='italic')
    ax.text(2.5, 0.97, 'SLM', transform=ax.get_xaxis_transform(),
            ha='center', va='top', fontsize=9, color='gray', style='italic')

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Composite Score (1–5)', fontsize=11)
    ax.set_title(
        'Human Evaluation: Composite Scores by Model and Approach\n(mean over 100 items, averaged across raters)',
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_ORDER, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'results_plot.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Exported {out}")


# ---------------------------------------------------------------------------
# Figure 2 — Category plot (composite scores by test-set category)
# ---------------------------------------------------------------------------
def plot_category(cat_means, plots_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping category_plot.pdf")
        return

    # Sort categories by descending composite score
    sorted_cats = sorted(cat_means.items(), key=lambda kv: kv[1], reverse=True)
    categories  = [kv[0] for kv in sorted_cats]
    scores      = [kv[1] for kv in sorted_cats]

    x   = np.arange(len(categories))
    fig, ax = plt.subplots(figsize=(8, 4))

    bars = ax.bar(
        x, scores, 0.55,
        color=PRIMARY_COLOR,
        alpha=0.85,
        edgecolor='black',
        linewidth=0.5,
    )
    for bar, s in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f'{s:.2f}',
            ha='center', va='bottom', fontsize=9,
        )

    ax.set_xlabel('Test-Set Category', fontsize=11)
    ax.set_ylabel('Composite Score (1–5)', fontsize=11)
    ax.set_title(
        'Composite Scores by Test-Set Category\n(all runs pooled, n=100)',
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'category_plot.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Exported {out}")


# ---------------------------------------------------------------------------
# Figure 3 — Latency plot
# ---------------------------------------------------------------------------
def plot_latency(row_by_key, plots_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping latency_plot.pdf")
        return

    width   = 0.25
    offsets = [-width, 0, width]
    x       = np.arange(len(MODELS_ORDER))

    fig, ax = plt.subplots(figsize=(8, 4))

    for i, approach in enumerate(APPROACHES):
        medians = []
        for model in MODELS_ORDER:
            r = row_by_key.get((model, approach))
            medians.append(r['median'] if r else 0.0)
        ax.bar(
            x + offsets[i], medians, width,
            label=approach,
            color=APPROACH_COLORS[approach],
            alpha=0.85,
            edgecolor='black',
            linewidth=0.5,
        )

    ax.set_xlabel('Model', fontsize=11)
    ax.set_ylabel('Median Latency (s)', fontsize=11)
    ax.set_title(
        'Inference Latency by Model and Configuration\n(median over 100 items)',
        fontsize=11,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(MODELS_ORDER, fontsize=9)
    ax.legend(fontsize=9)
    ax.set_ylim(bottom=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    out = os.path.join(plots_dir, 'latency_plot.pdf')
    fig.savefig(out, bbox_inches='tight')
    plt.close()
    print(f"Exported {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    root_dir  = os.path.dirname(os.path.abspath(__file__))
    plots_dir = os.path.join(root_dir, 'paper', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    kappa_path = os.path.join(root_dir, 'outputs', 'kappa_evaluation.json')
    eval_path  = os.path.join(root_dir, 'outputs', 'evaluation.json')
    cross_path = os.path.join(root_dir, 'outputs', 'cross_model_evaluation.json')

    # -------------------------------------------------------------------------
    # PART 1: EXTERNAL RATERS VS AUTHORS (Generalisability Validation)
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("External vs Author Generalisability Tests")
    print("=" * 50)

    external_stats_tex = []
    author_evals = defaultdict(dict)
    metrics = []

    if os.path.exists(kappa_path) and os.path.exists(eval_path):
        kappa_data = load_data(kappa_path)
        human_data = load_data(eval_path)

        for item in human_data:
            if item['evaluator'] in ['avilol', 'Arihant']:
                author_evals[(item['run_id'], item['item_index'])][item['evaluator']] = item['scores']

        y_ext_avilol   = defaultdict(list); y_avilol   = defaultdict(list)
        y_ext_arihant  = defaultdict(list); y_arihant  = defaultdict(list)
        y_ext_pooled   = defaultdict(list); y_author_pooled = defaultdict(list)
        _metrics_set   = set()

        for item in kappa_data:
            run_id, item_idx = item['run_id'], item['item_index']
            ext_scores = item['scores']
            authors    = author_evals.get((run_id, item_idx), {})

            for metric, score in ext_scores.items():
                _metrics_set.add(metric)
                authors_avg = []
                if 'avilol' in authors:
                    y_ext_avilol[metric].append(score)
                    y_avilol[metric].append(authors['avilol'][metric])
                    authors_avg.append(authors['avilol'][metric])
                if 'Arihant' in authors:
                    y_ext_arihant[metric].append(score)
                    y_arihant[metric].append(authors['Arihant'][metric])
                    authors_avg.append(authors['Arihant'][metric])
                if authors_avg:
                    y_ext_pooled[metric].append(score)
                    y_author_pooled[metric].append(sum(authors_avg) / len(authors_avg))

        metrics = sorted(list(_metrics_set))
        avg_kappa_scores = {}
        spearman_scores  = {}
        wilcoxon_pvals   = {}

        for metric in metrics:
            k1 = cohen_kappa_score_quadratic(y_ext_avilol[metric], y_avilol[metric]) \
                 if y_ext_avilol[metric] else 0
            k2 = cohen_kappa_score_quadratic(y_ext_arihant[metric], y_arihant[metric]) \
                 if y_ext_arihant[metric] else 0
            avg_kappa_scores[metric] = (k1 + k2) / 2.0

            s_corr, _ = spearmanr(y_ext_pooled[metric], y_author_pooled[metric])
            try:
                _, w_p = wilcoxon(y_ext_pooled[metric], y_author_pooled[metric])
            except ValueError:
                w_p = 1.0
            spearman_scores[metric] = s_corr
            wilcoxon_pvals[metric]  = w_p

        external_stats_tex.extend([
            "\\begin{table*}[h]", "\\centering",
            "\\begin{tabular}{lccc}", "\\toprule",
            "Dimension & Author-Ext $\\kappa$ & Spearman $r_s$ & Wilcoxon $p$-value \\\\",
            "\\midrule",
        ])
        for metric in metrics:
            kappa    = avg_kappa_scores.get(metric, 0.0)
            rho      = spearman_scores.get(metric, 0.0)
            pval     = wilcoxon_pvals.get(metric, 1.0)
            pval_str = "<0.001" if pval < 0.001 else f"{pval:.3f}"
            external_stats_tex.append(
                f"{metric.replace('_', ' ').title()} & {kappa:.2f} & {rho:.2f} & {pval_str} \\\\"
            )
        external_stats_tex.extend([
            "\\bottomrule", "\\end{tabular}",
            "\\caption{Statistical Generalisability: External Raters vs Author Consensus "
            "($N=200$ subset). Spearman $r_s$ assesses rank correlation, whilst the paired "
            "Wilcoxon test checks for systematic baseline deviations between external humans "
            "and authors.}",
            "\\label{tab:author_ext_stats}", "\\end{table*}\n",
        ])
        tex_path = os.path.join(root_dir, 'paper', 'latex', 'author_ext_stats.tex')
        with open(tex_path, 'w') as f:
            f.write("\n".join(external_stats_tex))
        print(f"Exported {tex_path}")
        for m in metrics:
            print(f"  {m} | rho: {spearman_scores[m]:.3f} | pval: {wilcoxon_pvals[m]:.3f}")

    # -------------------------------------------------------------------------
    # PART 2: LLM-as-Judge VS AUTHORS (Automated Reliability Test)
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("LLM-as-Judge vs Human Inference Reliability")
    print("=" * 50)

    if os.path.exists(cross_path) and os.path.exists(eval_path):
        model_data = load_data(cross_path)

        author_item_means = defaultdict(dict)
        for (rid, idx) in set(author_evals.keys()):
            for metric in metrics:
                vals = [author_evals[(rid, idx)][a][metric] for a in author_evals[(rid, idx)]]
                if vals:
                    author_item_means[(rid, idx)][metric] = sum(vals) / len(vals)

        self_eval  = []; other_eval = []
        h_overall  = defaultdict(list); m_overall = defaultdict(list)
        y_llm_paired   = defaultdict(list); y_human_paired = defaultdict(list)

        for m in model_data:
            rid, idx    = m['run_id'], m['item_index']
            scores      = m['scores']
            evaluator   = m['evaluator'].lower()
            target_base = m.get('target_base', '').lower()

            comp = sum(scores.values()) / len(scores)
            if evaluator == target_base and evaluator != "":
                self_eval.append(comp)
            else:
                other_eval.append(comp)

            h_match = author_item_means.get((rid, idx), {})
            if h_match:
                for k, v in scores.items():
                    y_llm_paired[k].append(v)
                    y_human_paired[k].append(h_match[k])
                    m_overall[k].append(v)
                    h_overall[k].append(h_match[k])

        llm_spearman = {}; llm_wilcoxon = {}; llm_inflation = {}
        for k in metrics:
            h_mean = sum(h_overall[k]) / len(h_overall[k]) if h_overall[k] else 0.0
            m_mean = sum(m_overall[k]) / len(m_overall[k]) if m_overall[k] else 0.0
            llm_inflation[k] = m_mean - h_mean
            if y_llm_paired[k] and y_human_paired[k]:
                c, _ = spearmanr(y_llm_paired[k], y_human_paired[k])
                llm_spearman[k] = c
                try:
                    _, wp = wilcoxon(y_llm_paired[k], y_human_paired[k])
                except ValueError:
                    wp = 1.0
                llm_wilcoxon[k] = wp

        overall_inf = sum(llm_inflation.values()) / len(llm_inflation)
        self_bias   = (
            (sum(self_eval) / len(self_eval)) - (sum(other_eval) / len(other_eval))
            if self_eval and other_eval else 0.0
        )

        bias_tex = [
            "\\begin{table*}[h]", "\\centering", "\\small",
            "\\begin{tabular}{lccc}", "\\toprule",
            "Dimension & $\\Delta$ (Inflation) & Spearman $r_s$ & Wilcoxon $p$-value \\\\",
            "\\midrule",
        ]
        for k in metrics:
            inf    = llm_inflation[k]
            rho    = llm_spearman[k]
            wp     = llm_wilcoxon[k]
            wp_str = "<0.001" if wp < 0.001 else f"{wp:.3f}"
            mark   = f"+{inf:.2f}" if inf > 0 else f"{inf:.2f}"
            bias_tex.append(
                f"{k.replace('_', ' ').title()} & {mark} & {rho:.2f} & {wp_str} \\\\"
            )
        bias_tex.extend([
            "\\midrule",
            f"\\textbf{{Composite}} & \\textbf{{+{overall_inf:.2f}}} & - & - \\\\",
            "\\bottomrule", "\\end{tabular}",
            "\\caption{Automated LLM-as-Judge bias statistics and rank coordination relative "
            "to Human truth.}",
            "\\label{tab:llm_bias}", "\\end{table*}\n",
        ])
        tex_path = os.path.join(root_dir, 'paper', 'latex', 'llm_judge_bias.tex')
        with open(tex_path, 'w') as f:
            f.write("\n".join(bias_tex))

        stats_out = {"self_bias": round(self_bias, 2), "overall_inflation": round(overall_inf, 2)}
        with open(os.path.join(root_dir, 'llm_judge_stats.json'), 'w') as f:
            json.dump(stats_out, f, indent=2)

        print(f"Exported {tex_path}")
        print(f"Self-Bias: {self_bias:.2f}, Composite Inflation: {overall_inf:.2f}")

    # -------------------------------------------------------------------------
    # PART 3: INFERENCE LATENCY ANALYSIS
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("Inference Latency Analysis")
    print("=" * 50)

    TABLE_ORDER = [
        ('Grok 4.1 Fast', 'Baseline'),     ('Grok 4.1 Fast', 'Task-focused'),
        ('Qwen-3 235B',   'Baseline'),     ('Qwen-3 235B',   'Task-focused'),
        ('SmolLM3-3B',    'Baseline'),     ('SmolLM3-3B',    'Task-focused'), ('SmolLM3-3B',  'Fine-tuned'),
        ('LFM2-700M',     'Baseline'),     ('LFM2-700M',     'Task-focused'), ('LFM2-700M',   'Fine-tuned'),
    ]

    latency_rows = []
    for run_key, (model_name, approach, model_class) in RUN_META.items():
        fpath = os.path.join(root_dir, 'outputs', f'{run_key}.json')
        if not os.path.exists(fpath):
            continue
        data  = load_data(fpath)
        lats  = [d['Latency'] for d in data if 'Latency' in d]
        s1s   = [d['Step1_Latency'] for d in data if 'Step1_Latency' in d]
        s2s   = [d['Latency'] - d['Step1_Latency']
                 for d in data if 'Latency' in d and 'Step1_Latency' in d]

        latency_rows.append({
            'model': model_name, 'approach': approach, 'class': model_class,
            'mean':    statistics.mean(lats),
            'std':     statistics.stdev(lats) if len(lats) > 1 else 0.0,
            'median':  statistics.median(lats),
            's1_mean': statistics.mean(s1s) if s1s else None,
            's2_mean': statistics.mean(s2s) if s2s else None,
        })
        print(
            f"  {model_name} / {approach}: "
            f"mean={latency_rows[-1]['mean']:.2f}s  "
            f"std={latency_rows[-1]['std']:.2f}s  "
            f"median={latency_rows[-1]['median']:.2f}s"
        )

    row_by_key = {(r['model'], r['approach']): r for r in latency_rows}

    lat_tex = [
        "\\begin{table*}[t]", "\\centering", "\\small",
        "\\caption{Inference latency per configuration ($n=100$ items each). "
        "For two-phase approaches (Task-focused, Fine-tuned), Step~1 and Step~2 "
        "denote the misconception-identification and Socratic-generation passes "
        "respectively, with the total latency being their sum. "
        "Grok Baseline mean is inflated by a small number of high-latency API "
        "outliers (max~570\\,s); the median is a more representative central "
        "tendency for that configuration.}",
        "\\label{tab:latency}",
        "\\begin{tabular}{@{}llcccc@{}}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Approach} & "
        "\\textbf{Mean (s)} & \\textbf{(Std)} & "
        "\\textbf{Median (s)} & \\textbf{Step~1 / Step~2 (s)} \\\\",
        "\\midrule",
    ]

    prev_model = None
    for model_name, approach in TABLE_ORDER:
        row = row_by_key.get((model_name, approach))
        if row is None:
            continue
        if prev_model is not None and prev_model != model_name:
            lat_tex.append("\\midrule")
        prev_model  = model_name
        model_str   = model_name if approach == 'Baseline' else ''
        step_str    = (f"{row['s1_mean']:.2f} / {row['s2_mean']:.2f}"
                       if row['s1_mean'] is not None else "---")
        lat_tex.append(
            f"{model_str} & {approach} & "
            f"{row['mean']:.2f} & ({row['std']:.2f}) & "
            f"{row['median']:.2f} & {step_str} \\\\"
        )

    lat_tex += ["\\bottomrule", "\\end{tabular}", "\\end{table*}\n"]

    tex_path = os.path.join(root_dir, 'paper', 'latex', 'latency_stats.tex')
    with open(tex_path, 'w') as f:
        f.write("\n".join(lat_tex))
    print(f"Exported {tex_path}")

    # -------------------------------------------------------------------------
    # PART 4: GENERATE ALL PLOTS
    # -------------------------------------------------------------------------
    print("=" * 50)
    print("Generating plots")
    print("=" * 50)

    # Load evaluation data and compute scores
    if os.path.exists(eval_path):
        eval_data         = load_data(eval_path)
        item_avg, run_means = compute_eval_scores(eval_data)
        cat_means         = compute_category_scores(item_avg, root_dir)

        print("Run means:")
        for rid, mean in sorted(run_means.items()):
            print(f"  {rid}: {mean:.3f}")
        print("Category means:")
        for cat, mean in sorted(cat_means.items(), key=lambda kv: -kv[1]):
            print(f"  {cat}: {mean:.3f}")

        plot_results(run_means, plots_dir)
        plot_category(cat_means, plots_dir)

    plot_latency(row_by_key, plots_dir)


if __name__ == "__main__":
    main()
