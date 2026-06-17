import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import os

base = r'E:\paper\new\SecureProtoCoT\outputs\experiment_isr'

dfs = {}
for cfg in ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3']:
    df = pd.read_csv(os.path.join(base, f'iterations_{cfg}.csv'))
    dfs[cfg] = df

configs = ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3']
colors = {'ISR-0': '#9e9e9e', 'ISR-1': '#1976d2', 'ISR-2': '#e65100', 'ISR-3': '#2e7d32'}
lstyles = {'ISR-0': ':', 'ISR-1': '--', 'ISR-2': '-.', 'ISR-3': '-'}
labels = {'ISR-0': 'ISR-0 (no feedback)',
          'ISR-1': 'ISR-1 (generic)',
          'ISR-2': 'ISR-2 (attention)',
          'ISR-3': 'ISR-3 (attention + spec)'}

all_prompts = sorted(dfs['ISR-0']['prompt_id'].unique())

best_sofar = {}
for cfg in configs:
    df = dfs[cfg]
    per_prompt = {}
    for pid in all_prompts:
        rows = df[df['prompt_id'] == pid].sort_values('iteration')
        if len(rows) == 0:
            continue
        pvals = rows['p_vul'].values
        iters = rows['iteration'].values
        bsf = np.minimum.accumulate(pvals)
        per_prompt[pid] = (iters, bsf)
    best_sofar[cfg] = per_prompt

def align_trajectory(iters, bsf, max_iter=4):
    result = np.full(max_iter + 1, np.nan)
    for i, v in zip(iters, bsf):
        if i <= max_iter:
            result[i] = v
    last_valid = np.nan
    for i in range(max_iter + 1):
        if not np.isnan(result[i]):
            last_valid = result[i]
        else:
            result[i] = last_valid
    return result

traj_mean, traj_sem = {}, {}
for cfg in configs:
    padded = np.array([align_trajectory(*best_sofar[cfg][p]) for p in all_prompts])
    traj_mean[cfg] = np.mean(padded, axis=0)
    traj_sem[cfg] = np.std(padded, axis=0, ddof=1) / np.sqrt(padded.shape[0])

improvements = {}
for pid in all_prompts:
    it, bsf = best_sofar['ISR-3'][pid]
    if len(bsf) >= 2 and bsf[0] > 0.0005:
        ratio = bsf[0] / max(bsf[-1], 1e-9)
        improvements[pid] = ratio
top3 = sorted(improvements, key=improvements.get, reverse=True)[:3]

# ---- FIGURE: original 2-panel design ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# --- LEFT: Mean best-so-far across all 15 prompts ---
for cfg in configs:
    m = traj_mean[cfg]
    s = traj_sem[cfg]
    ax1.fill_between(range(5), np.maximum(m - s, 1e-8), m + s,
                     color=colors[cfg], alpha=0.1)
    ax1.plot(range(5), m, color=colors[cfg], linestyle=lstyles[cfg],
             linewidth=2.0 if cfg == 'ISR-3' else 1.4,
             marker='D' if cfg == 'ISR-2' else 'o', markersize=6,
             markerfacecolor='white', markeredgewidth=1.5,
             label=labels[cfg], zorder=5 if cfg == 'ISR-3' else 3)

ax1.set_yscale('log')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Best-so-far P(vul)  [log scale]', fontsize=11)
ax1.legend(frameon=True, fontsize=7.8, loc='upper right', ncol=2)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xlim(-0.1, 4.2)
ax1.grid(True, alpha=0.2, which='both')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.tick_params(labelsize=9)
ax1.set_title('A  Iterative safety refinement: best-so-far P(vul)', fontsize=11.5,
              fontweight='bold', loc='left')

drop_pct = (1 - traj_mean['ISR-3'][4] / traj_mean['ISR-3'][0]) * 100
ax1.annotate(f'ISR-3: {drop_pct:.0f}% drop from iter 0 to 4',
             xy=(4, traj_mean['ISR-3'][4]),
             xytext=(2.4, traj_mean['ISR-3'][4] * 50),
             fontsize=7.5, color=colors['ISR-3'], fontweight='bold',
             arrowprops=dict(arrowstyle='->', color=colors['ISR-3'], lw=0.9))

# --- RIGHT: Individual case trajectories ---
case_labels = {
    'P11_struct_copy': 'P11 (struct deep copy)',
    'P05_free_memory': 'P05 (memory free)',
    'P09_memcpy_wrapper': 'P09 (memcpy wrapper)',
}
markers_case = {'ISR-1': 's', 'ISR-2': '^', 'ISR-3': 'o'}

for pid in top3:
    for cfg in ['ISR-1', 'ISR-2', 'ISR-3']:
        if pid not in best_sofar[cfg]:
            continue
        it, bsf = best_sofar[cfg][pid]
        alpha = 1.0 if cfg == 'ISR-3' else 0.45
        lw = 1.6 if cfg == 'ISR-3' else 0.9
        ax2.plot(it, bsf, color=colors[cfg], linestyle=lstyles[cfg],
                 linewidth=lw, marker=markers_case[cfg], markersize=5.5,
                 markerfacecolor='white', markeredgewidth=1.2,
                 alpha=alpha,
                 label=f'{case_labels[pid]} [{cfg}]' if cfg == 'ISR-3' else
                       f'{case_labels[pid]} [{cfg}]')

ax2.set_yscale('log')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Best-so-far P(vul)  [log scale]', fontsize=11)
ax2.legend(frameon=True, fontsize=6.5, loc='upper right', ncol=1)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xlim(-0.1, 4.2)
ax2.grid(True, alpha=0.2, which='both')
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.tick_params(labelsize=9)
ax2.set_title('B  Selected prompt trajectories', fontsize=11.5,
              fontweight='bold', loc='left')

plt.tight_layout(pad=1.5)

# Save as fig_trajectory0
outpath_pdf = r'E:\paper\new\ese\LaTeX_DL_468198_240419\fig_trajectory0.pdf'
outpath_png = r'E:\paper\new\ese\LaTeX_DL_468198_240419\fig_trajectory0.png'
plt.savefig(outpath_pdf, dpi=300, bbox_inches='tight')
plt.savefig(outpath_png, dpi=200, bbox_inches='tight')
print(f'Saved: {outpath_pdf}')
print(f'Saved: {outpath_png}')
