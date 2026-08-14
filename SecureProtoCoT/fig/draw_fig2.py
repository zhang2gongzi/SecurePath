"""
Fig 2: Cross-model human evaluation comparison (grouped bar chart).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# --- Data ---
configs = ['ISR-0', 'ISR-1', 'ISR-2', 'ISR-3', 'B4', 'B5', 'B6', 'B7']
deepseek = [45.5, 72.7, 80.0, 87.5, 71.4, 78.6, 83.3, 76.7]
glm      = [14.3, 50.0, 61.5, 68.4, 53.8, 53.8, 46.2, 61.5]
sonnet   = [40.0, 66.7, 73.3, 80.0, 86.7, 93.3, 86.7, 86.7]

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 5.5))

x = np.arange(len(configs))
width = 0.25

bars1 = ax.bar(x - width, deepseek, width, color='#4472C4', edgecolor='white', label='DeepSeek-v4-pro')
bars2 = ax.bar(x,         glm,      width, color='#ED7D31', edgecolor='white', label='GLM-5.1')
bars3 = ax.bar(x + width, sonnet,   width, color='#70AD47', edgecolor='white', label='Sonnet-4-6')

# Value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 1, f'{h:.1f}%',
                ha='center', va='bottom', fontsize=7)

# ISR group separator
ax.axvline(x=3.5, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax.text(1.75, 102, 'ISR Ablations', ha='center', fontsize=9, fontweight='bold', color='#333')
ax.text(5.75, 102, 'Baselines', ha='center', fontsize=9, fontweight='bold', color='#333')

ax.set_xticks(x)
ax.set_xticklabels(configs, fontsize=9)
ax.set_ylabel('Human-Evaluated SAFE Rate (%)', fontsize=10)
ax.set_ylim(0, 108)
ax.legend(fontsize=9, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
fig.savefig('cross_model_comparison.pdf', dpi=200, bbox_inches='tight')
fig.savefig('cross_model_comparison.png', dpi=200, bbox_inches='tight')
print("Saved: cross_model_comparison.pdf / .png")
