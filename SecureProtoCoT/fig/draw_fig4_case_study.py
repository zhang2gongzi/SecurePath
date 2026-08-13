"""
Fig 4: ISR-random Case Study — P09 memcpy_wrapper
Three-panel code comparison: (a) Initial, (b) ISR-2, (c) ISR-random
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches
import numpy as np

fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(14, 5.2))

# --- Colors (light theme, like VS Code GitHub Light) ---
C_CODE_BG = '#F6F8FA'
C_BORDER = '#D0D7DE'
C_LINE_NO = '#8B949E'
C_CODE = '#1F2328'
C_KEYWORD = '#CF222E'     # void, if, size_t
C_FUNC = '#8250DF'         # memcpy, memmove
C_COMMENT = '#6E7781'
C_GREEN_BG = '#DAFBE1'
C_GREEN_BORDER = '#2DA44E'
C_RED_BG = '#FFEBE9'
C_RED_BORDER = '#CF222E'
C_RED_HIGHLIGHT = '#FFEBE9'

def parse_tokens(line):
    """Naive C syntax highlighting: return list of (text, color) tuples."""
    tokens = []
    keywords = {'void', 'if', 'return', 'const', 'size_t', 'include'}
    unsafe_funcs = {'memcpy', 'strcpy', 'sprintf', 'gets', 'strcat'}
    safe_funcs = {'memmove', 'strncpy', 'snprintf', 'fgets', 'strncat'}

    i = 0
    while i < len(line):
        if line[i].isspace():
            j = i
            while j < len(line) and line[j].isspace():
                j += 1
            tokens.append((line[i:j], C_CODE))
            i = j
        elif line[i] == '#' and (i == 0 or line[:i].isspace()):
            j = i
            while j < len(line) and line[j] != '\n':
                j += 1
            tokens.append((line[i:j], C_COMMENT))
            i = j
        elif line[i].isalpha() or line[i] == '_':
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == '_'):
                j += 1
            word = line[i:j]
            if word in keywords:
                tokens.append((word, C_KEYWORD))
            elif word in unsafe_funcs:
                tokens.append((word, C_FUNC))
            elif word in safe_funcs:
                tokens.append((word, C_FUNC))
            else:
                tokens.append((word, C_CODE))
            i = j
        else:
            tokens.append((line[i], C_CODE))
            i += 1
    return tokens

def draw_code_panel(ax, code_lines, title, subtitle, highlight_lines=None, badge_text=None, badge_color=C_GREEN_BORDER):
    """Draw a single code panel with syntax highlighting."""
    n = len(code_lines)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n + 1.5)
    ax.axis('off')

    # Title + subtitle
    ax.text(5, n + 1.1, title, ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1F2328')
    ax.text(5, n + 0.5, subtitle, ha='center', va='center',
            fontsize=8.5, color=C_LINE_NO)

    # Code background
    rect = FancyBboxPatch((0.2, -0.2), 9.6, n + 0.5,
                          boxstyle="round,pad=0.1", facecolor=C_CODE_BG,
                          edgecolor=C_BORDER, linewidth=1.0)
    ax.add_patch(rect)

    for i, line in enumerate(code_lines):
        y = n - i - 0.1

        # Line number
        ax.text(0.5, y, str(i + 1), ha='right', va='center',
                fontfamily='monospace', fontsize=10, color=C_LINE_NO)

        # Highlight background
        if highlight_lines and i in highlight_lines:
            hl = highlight_lines[i]
            hl_rect = FancyBboxPatch(
                (1.0, y - 0.32), 8.6, 0.62,
                boxstyle="round,pad=0.05",
                facecolor=hl.get('bg', C_RED_HIGHLIGHT),
                edgecolor=hl.get('border', C_RED_BORDER),
                linewidth=1.0, alpha=0.6
            )
            ax.add_patch(hl_rect)

        # Draw tokens with syntax highlighting
        tokens = parse_tokens(line)
        x_pos = 1.2
        for text, color in tokens:
            ax.text(x_pos, y, text, ha='left', va='center',
                    fontfamily='monospace', fontsize=10, color=color)
            # Approximate text width
            x_pos += len(text) * 0.06

        # Highlight annotation arrow
        if highlight_lines and i in highlight_lines and 'anno' in highlight_lines[i]:
            anno = highlight_lines[i]['anno']
            ax.annotate(anno.get('text', ''),
                xy=(anno.get('x', 5), y - 0.45),
                xytext=(anno.get('x', 5), y - 1.2),
                fontsize=7.5, fontweight='bold', color=anno.get('color', C_RED_BORDER),
                ha='center', va='top',
                arrowprops=dict(arrowstyle='->', color=anno.get('color', C_RED_BORDER), lw=1.5))

    # Badge
    if badge_text:
        badge_y = -0.5
        badge = FancyBboxPatch((2.5, badge_y), 5.0, 0.55,
                               boxstyle="round,pad=0.08",
                               facecolor=badge_color + '18',
                               edgecolor=badge_color, linewidth=1.5)
        ax.add_patch(badge)
        ax.text(5, badge_y + 0.27, badge_text, ha='center', va='center',
                fontsize=9, fontweight='bold', color=badge_color)

# ============================================
# Panel (a): Initial code
# ============================================
initial_code = [
    '#include <string.h>',
    '',
    'void safe_memcpy(void *dst, const void *src, size_t n) {',
    '    if (!dst || !src) return;',
    '    memcpy(dst, src, n);',
    '}',
]

init_highlights = {
    4: {'bg': '#FFF3CD', 'border': '#D4A017',
        'anno': {'text': 'memcpy — overlap\nrisk (CWE-787)', 'x': 4.5, 'color': '#D4A017'}},
}

draw_code_panel(ax_a, initial_code,
                '(a) Initial generation',
                'P09 memcpy_wrapper — LLM baseline output',
                highlight_lines=init_highlights,
                badge_text='memcpy present  ✗  VULNERABLE',
                badge_color=C_RED_BORDER)

# ============================================
# Panel (b): ISR-2 (real attention)
# ============================================
isr2_code = [
    '#include <string.h>',
    '',
    'void safe_memcpy(void *dst, const void *src, size_t n) {',
    '    if (!dst || !src) return;',
    '    memmove(dst, src, n);',
    '}',
]

isr2_highlights = {
    4: {'bg': C_GREEN_BG, 'border': C_GREEN_BORDER,
        'anno': {'text': 'memmove — overlap-safe\nattn=0.92, CWE-787', 'x': 4.8, 'color': C_GREEN_BORDER}},
}

draw_code_panel(ax_b, isr2_code,
                '(b) ISR-2: Real attention',
                'Classifier flags memcpy → LLM replaces with memmove',
                highlight_lines=isr2_highlights,
                badge_text='memmove replaces memcpy  ✓  SAFE',
                badge_color=C_GREEN_BORDER)

# ============================================
# Panel (c): ISR-random
# ============================================
isr_random_code = [
    '#include <string.h>',
    '#include <stddef.h>',
    '',
    'void safe_memcpy(void *dst, const void *src, size_t n) {',
    '    if (!dst || !src) return;',
    '    if (n == 0) return;',
    '    memcpy(dst, src, n);',
    '}',
]

isr_random_highlights = {
    3: {'bg': '#FFF3CD', 'border': '#D4A017',
        'anno': {'text': 'random flag:\n"size_t n"', 'x': 7.5, 'color': '#D4A017'}},
    5: {'bg': '#FFF3CD', 'border': '#D4A017',
        'anno': {'text': 'random flag:\n"early return"', 'x': 7.0, 'color': '#D4A017'}},
    6: {'bg': C_RED_HIGHLIGHT, 'border': C_RED_BORDER,
        'anno': {'text': 'memcpy survives\n— NOT flagged', 'x': 3.5, 'color': C_RED_BORDER}},
}

draw_code_panel(ax_c, isr_random_code,
                '(c) ISR-random: Random attention',
                'Random flags param type & return → cosmetic fix only',
                highlight_lines=isr_random_highlights,
                badge_text='memcpy survives  ✗  UNSAFE',
                badge_color=C_RED_BORDER)

# --- Legend at bottom ---
legend_elements = [
    mpatches.Patch(facecolor=C_GREEN_BG, edgecolor=C_GREEN_BORDER, label='ISR-2: real attention fix'),
    mpatches.Patch(facecolor=C_RED_HIGHLIGHT, edgecolor=C_RED_BORDER, label='ISR-random: vulnerability missed'),
    mpatches.Patch(facecolor='#FFF3CD', edgecolor='#D4A017', label='Random feedback target'),
]
fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=8, framealpha=0.9)

plt.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig('fig/fig4_isr_random_case_study.pdf', dpi=200, bbox_inches='tight')
fig.savefig('fig/fig4_isr_random_case_study.png', dpi=200, bbox_inches='tight')
print("Saved: fig/fig4_isr_random_case_study.pdf / .png")
