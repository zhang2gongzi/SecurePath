import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 4.2))

# --- Color scheme ---
c_bad  = '#c62828'  # existing paradigm (red tone)
c_good = '#2e7d32'  # ISR (green tone)
c_box  = '#f5f5f5'
c_edge_bad = '#b71c1c'
c_edge_good = '#1b5e20'
c_arrow_bad = '#d32f2f'
c_arrow_good = '#388e3c'
c_fail = '#e53935'
c_pass = '#43a047'
c_text = '#333333'

def draw_box(ax, x, y, w, h, text, color, edgecolor, fontsize=8.5, bold=False):
    """Draw a rounded box with text."""
    weight = 'bold' if bold else 'normal'
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor=edgecolor, linewidth=1.5)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=weight, color=c_text)

def draw_arrow(ax, x1, y1, x2, y2, color, lw=1.8):
    """Draw arrow from (x1,y1) to (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                connectionstyle='arc3,rad=0'))

def draw_loop_arrow(ax, x, y, radius, color, lw=1.8):
    """Draw a circular loop arrow."""
    import matplotlib.patches as patches
    arc = patches.Arc((x, y), radius*2, radius*2, angle=0, theta1=30, theta2=330,
                      color=color, lw=lw)
    ax.add_patch(arc)
    # arrowhead
    tip_x = x + radius * np.cos(np.radians(30))
    tip_y = y + radius * np.sin(np.radians(30))
    ax.annotate('', xy=(tip_x, tip_y), xytext=(tip_x - 0.06, tip_y + 0.08),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw))

# ===== LEFT: Existing Paradigm =====
ax_left.set_xlim(0, 4)
ax_left.set_ylim(0, 4.5)
ax_left.axis('off')
ax_left.set_title('Existing Paradigm', fontsize=12, fontweight='bold',
                  color=c_bad, pad=8)

# LLM box
draw_box(ax_left, 1.2, 3.5, 1.6, 0.55, 'LLM', c_box, c_edge_bad, fontsize=9.5, bold=True)
# Code box
draw_box(ax_left, 1.2, 2.5, 1.6, 0.55, 'Generated Code', c_box, c_edge_bad)
# Self-assessment box
draw_box(ax_left, 1.2, 1.5, 1.6, 0.55, 'LLM Self-Assessment', c_box, c_edge_bad)

# Arrows: LLM -> Code -> Self-assessment
draw_arrow(ax_left, 2.0, 3.5, 2.0, 3.1, c_arrow_bad)
draw_arrow(ax_left, 2.0, 2.5, 2.0, 2.1, c_arrow_bad)

# Loop back: Self-assessment -> LLM (circular dependency)
ax_left.annotate('', xy=(0.9, 3.78), xytext=(0.9, 1.78),
                arrowprops=dict(arrowstyle='->', color=c_fail, lw=2.2,
                                connectionstyle='arc3,rad=-0.55'))
ax_left.text(0.18, 2.78, 'circular\ndependency', fontsize=7, color=c_fail,
             ha='center', va='center', fontstyle='italic')

# Result: unsafe
ax_left.text(2.0, 1.05, 'Unsafe Code', fontsize=10, fontweight='bold',
             color=c_fail, ha='center')
ax_left.text(2.0, 0.75, '"the same model evaluates itself"', fontsize=7.5,
             color='#888888', ha='center', fontstyle='italic')

# Cross mark
ax_left.plot(2.0, 0.35, marker='$\\times$', markersize=28, color=c_fail,
             markeredgewidth=2)

# ===== RIGHT: ISR (Ours) =====
ax_right.set_xlim(0, 4)
ax_right.set_ylim(0, 4.5)
ax_right.axis('off')
ax_right.set_title('ISR (Ours)', fontsize=12, fontweight='bold',
                   color=c_good, pad=8)

# LLM box
draw_box(ax_right, 1.2, 3.5, 1.6, 0.55, 'LLM', c_box, c_edge_good, fontsize=9.5, bold=True)
# Code box
draw_box(ax_right, 1.2, 2.5, 1.6, 0.55, 'Generated Code', c_box, c_edge_good)

# External classifier (highlighted differently)
draw_box(ax_right, 0.3, 1.5, 1.6, 0.55, 'External\nSafety Classifier',
         '#e8f5e9', c_edge_good, fontsize=8.5, bold=True)

# Feedback box
draw_box(ax_right, 2.1, 1.5, 1.6, 0.55, 'Attention-Guided\nFeedback', c_box, c_edge_good, fontsize=8.5)

# Arrows: LLM -> Code -> Classifier
draw_arrow(ax_right, 2.0, 3.5, 2.0, 3.1, c_arrow_good)
draw_arrow(ax_right, 2.0, 2.5, 1.1, 2.1, c_arrow_good)

# Classifier -> Feedback
draw_arrow(ax_right, 1.9, 1.78, 2.1, 1.78, c_arrow_good)

# Feedback -> LLM (external signal)
ax_right.annotate('', xy=(2.9, 3.78), xytext=(2.9, 2.1),
                arrowprops=dict(arrowstyle='->', color=c_pass, lw=2.2,
                                connectionstyle='arc3,rad=0.55'))
ax_right.text(3.55, 2.78, 'external\nsignal', fontsize=7, color=c_pass,
              ha='center', va='center', fontstyle='italic')

# Iteration label
ax_right.text(3.1, 3.15, 'iterate', fontsize=6.5, color=c_good,
              ha='center', fontstyle='italic')

# Result: safe
ax_right.text(2.0, 1.05, 'Safe Code', fontsize=10, fontweight='bold',
              color=c_pass, ha='center')
ax_right.text(2.0, 0.75, '"independent classifier guides repair"', fontsize=7.5,
              color='#888888', ha='center', fontstyle='italic')

# Check mark
ax_right.plot(2.0, 0.35, marker='$\\checkmark$', markersize=28, color=c_pass,
              markeredgewidth=2)

# Panel labels
fig.text(0.02, 0.96, 'A', fontsize=14, fontweight='bold')
fig.text(0.52, 0.96, 'B', fontsize=14, fontweight='bold')

plt.tight_layout(pad=1.0)
fig.savefig(r'E:\paper\new\ese\LaTeX_DL_468198_240419\fig_paradigm_comparison.pdf',
            dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig(r'E:\paper\new\ese\LaTeX_DL_468198_240419\fig_paradigm_comparison.png',
            dpi=200, bbox_inches='tight', facecolor='white')
print('Done: fig_paradigm_comparison.pdf + fig_paradigm_comparison.png')
