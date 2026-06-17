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
          'ISR-1': 'ISR-1 (generic alert)',
          'ISR-2': 'ISR-2 (attention-guided)',
          'ISR-3': 'ISR-3 (attention + spec)'}

all_prompts = sorted(dfs['ISR-0']['prompt_id'].unique())

# --- best-so-far per prompt ---
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

# --- Panel A: mean best-so-far with forward-fill ---
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

# --- Panel B cases ---
# P11: 2 iters, ×139  |  P14: 3 iters, ×4.6  |  P07: 5 iters, ×12.1
case_specs = [
    ('P11_struct_copy',   'P11 (struct deep copy)',  '×139 in 1 iteration → converged',   'upper right'),
    ('P14_realloc_array', 'P14 (realloc array)',     '×4.6 in 2 iterations → converged',  'upper right'),
    ('P07_int_parse',     'P07 (integer parse)',     '×12.1 over 4 iterations',            'upper right'),
]
case_colors = ['#1565c0', '#6a1b9a', '#c62828']

# ---- FIGURE ----
fig = plt.figure(figsize=(11, 4.8))
gs = fig.add_gridspec(1, 2, wspace=0.32, left=0.07, right=0.97, top=0.86, bottom=0.15)
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])

# ===== PANEL A: Mean best-so-far =====
for cfg in configs:
    m = traj_mean[cfg]
    s = traj_sem[cfg]
    ax1.fill_between(range(5), np.maximum(m - s, 1e-8), m + s,
                     color=colors[cfg], alpha=0.1)
    ax1.plot(range(5), m, color=colors[cfg], linestyle=lstyles[cfg],
             linewidth=2.2 if cfg == 'ISR-3' else 1.5,
             marker='D' if cfg == 'ISR-2' else 'o', markersize=5.5,
             markerfacecolor='white', markeredgewidth=1.5,
             label=labels[cfg], zorder=5 if cfg == 'ISR-3' else 3)

ax1.set_yscale('log')
ax1.set_xlabel('Iteration', fontsize=11)
ax1.set_ylabel('Best-so-far P(vul)  [log scale]', fontsize=11)
ax1.set_xticks([0, 1, 2, 3, 4])
ax1.set_xlim(-0.1, 4.3)
ax1.grid(True, alpha=0.2, which='both')
ax1.yaxis.set_major_formatter(ScalarFormatter())
ax1.tick_params(labelsize=9)
ax1.set_title('A  Mean best-so-far P(vul) over 15 prompts', fontsize=11.5,
              fontweight='bold', loc='left')
ax1.legend(frameon=True, fontsize=7.5, loc='upper right', ncol=2,
           handlelength=1.8, handletextpad=0.4, columnspacing=0.6)

# ===== PANEL B: Individual ISR-3 cases =====
for idx, (pid, label, annotation, ann_loc) in enumerate(case_specs):
    iters, bsf = best_sofar['ISR-3'][pid]
    n_real = len(iters)
    color = case_colors[idx]

    # Plot actual data: solid line + filled markers
    ax2.plot(iters, bsf, color=color, linestyle='-', linewidth=2.0,
             marker='o', markersize=7, markerfacecolor=color, markeredgewidth=0,
             label=label, zorder=5)

    # Annotate near the last data point
    last_x, last_y = iters[-1], bsf[-1]
    ax2.annotate(annotation,
                 xy=(last_x, last_y),
                 xytext=(last_x + 0.3, last_y * (3.5 if idx == 0 else
                                                   7.0 if idx == 1 else 2.5)),
                 fontsize=7.5, color=color, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color=color, lw=1.0,
                                 connectionstyle='arc3,rad=0.2'),
                 va='center', ha='left')

ax2.set_yscale('log')
ax2.set_xlabel('Iteration', fontsize=11)
ax2.set_ylabel('Best-so-far P(vul)  [log scale]', fontsize=11)
ax2.set_xticks([0, 1, 2, 3, 4])
ax2.set_xlim(-0.1, 4.8)
ax2.grid(True, alpha=0.2, which='both')
ax2.yaxis.set_major_formatter(ScalarFormatter())
ax2.tick_params(labelsize=9)
ax2.set_title(r'B  ISR-3 trajectories: fast convergence vs. progressive refinement',
              fontsize=10.5, fontweight='bold', loc='left')
ax2.legend(frameon=True, fontsize=8.5, loc='upper right',
           handlelength=1.5, handletextpad=0.5)

# Footer note
fig.text(0.52, 0.03,
         'Panel B: markers show actual iteration points; trajectories end at convergence (no further P(vul) improvement).',
         fontsize=7.5, color='#555555', ha='center', style='italic')

plt.savefig(r'E:\paper\new\ese\LaTeX_DL_468198_240419\fig_trajectory.pdf',
            dpi=300, bbox_inches='tight')
plt.savefig(r'E:\paper\new\ese\LaTeX_DL_468198_240419\fig_trajectory.png',
            dpi=200, bbox_inches='tight')
print('Done: fig_trajectory.pdf + fig_trajectory.png')
