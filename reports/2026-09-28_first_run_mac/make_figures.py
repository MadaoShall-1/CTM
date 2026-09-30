"""Render the report figures (light + dark PNGs) from the small files in this folder.

Usage (from this folder, any Python with matplotlib):  python make_figures.py
Palette/marks follow the reference data-viz palette (validated for CVD in both modes).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
DPI = 144                       # 1 pt = 2 px
PX = 72 / DPI                   # points per pixel
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781',
                  grid='#e1e0d9', axis='#c3c2b7', series=['#2a78d6', '#eb6834', '#1baf7a']),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781',
                 grid='#2c2c2a', axis='#383835', series=['#3987e5', '#d95926', '#199e70']),
}
# Team pilot, EXPERIMENT_ANALYSIS_AND_SCENARIO_ISSUES.md §4 (6,000 steps, seeds 0-2, 100 test episodes).
TEAM_DIRECT_6000 = {'pdqn': 2.67, 'mpdqn': 5.67, 'her_pdqn': 3.00, 'her_mpdqn': 40.67}
NAMES = {'pdqn': 'P-DQN', 'mpdqn': 'MP-DQN', 'her_pdqn': 'HER-PDQN', 'her_mpdqn': 'HER-MPDQN'}


def style(ax, t, grid_axis='y'):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['axis'])
        ax.spines[side].set_linewidth(1 * PX)
    ax.tick_params(colors=t['muted'], labelcolor=t['muted'], width=1 * PX, length=3)
    ax.grid(axis=grid_axis, color=t['grid'], linewidth=1 * PX, linestyle='-')
    ax.set_axisbelow(True)


def figure(t, size=(7.2, 4.0)):
    fig, ax = plt.subplots(figsize=size, dpi=DPI)
    fig.patch.set_facecolor(t['surface'])
    style(ax, t)
    return fig, ax


def titles(fig, t, title, subtitle):
    fig.text(0.012, 0.965, title, color=t['ink'], fontsize=11.5, fontweight='bold', va='top')
    fig.text(0.012, 0.905, subtitle, color=t['ink2'], fontsize=8.5, va='top')


def legend(ax, t, handles, labels):
    ax.legend(handles, labels, loc='lower left', bbox_to_anchor=(0, 1.0), ncol=len(labels),
              frameon=False, fontsize=8.5, labelcolor=t['ink2'], handlelength=1.6,
              borderaxespad=0.3, columnspacing=1.4)


def save(fig, name, mode):
    fig.savefig(HERE / f'{name}_{mode}.png', dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def end_labels(ax, t, x_end, items, min_px=16):
    """Direct end labels, skipping any that would sit within min_px of one already placed.

    Call after axis limits/scales are final. The legend carries identity for skipped labels.
    """
    ax.figure.canvas.draw()
    placed = []
    for value, text in sorted(items):
        y_px = ax.transData.transform((x_end, value))[1]
        if all(abs(y_px - p) >= min_px for p in placed):
            ax.annotate(text, (x_end, value), xytext=(8, 0), textcoords='offset points',
                        va='center', fontsize=8.5, color=t['ink2'], annotation_clip=False)
            placed.append(y_px)


def ctwm_evaluation(mode):
    t = THEMES[mode]
    evaluations = [e for e in json.loads((HERE / 'ctwm_online_1h_evaluations.json').read_text())
                   if e['split'] in ('fixed', 'fixed_final')]
    x = [e['env_steps'] / 1000 for e in evaluations]
    fig, ax = figure(t)
    fig.subplots_adjust(left=0.085, right=0.84, top=0.74, bottom=0.14)
    series = [('success', 'Delivered'), ('pickup', 'Picked up'), ('oob', 'Out of bounds')]
    handles = []
    for (key, label), color in zip(series, t['series']):
        y = [100 * e[key] for e in evaluations]
        line, = ax.plot(x, y, color=color, linewidth=2 * PX, solid_joinstyle='round',
                        solid_capstyle='round', marker='o', markersize=8 * PX,
                        markeredgecolor=t['surface'], markeredgewidth=2 * PX, zorder=3, clip_on=False)
        handles.append(line)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, max(x) * 1.02 if max(x) else 1)
    ax.set_yticks(range(0, 101, 25))
    ax.set_yticklabels([f'{v}%' for v in range(0, 101, 25)])
    end_labels(ax, t, x[-1], [(100 * evaluations[-1][k], f'{lab} {100 * evaluations[-1][k]:.0f}%')
                              for k, lab in series])
    ax.set_xlabel('Environment steps (thousands)', color=t['ink2'], fontsize=8.5)
    legend(ax, t, handles, [label for _, label in series])
    titles(fig, t, 'CT-WM (NRSM + actor-critic): fixed test scenes during a 1-hour run',
           'Deterministic policy on the same 20 relay scenes (70000-70019) at each checkpoint; '
           'timeouts are the remainder.\nSeed 17, compact NRSM, Apple M4 CPU, 2026-09-28.')
    save(fig, 'ctwm_1h_fixed_scene_eval', mode)


def ctwm_gradient(mode):
    t = THEMES[mode]
    with (HERE / 'ctwm_online_1h_train_log.csv').open() as stream:
        rows = [r for r in csv.DictReader(stream) if r['model_gradient_norm']]
    x = [int(r['model_updates']) for r in rows]
    y = [float(r['model_gradient_norm']) for r in rows]
    fig, ax = figure(t)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.14)
    ax.plot(x, y, color=t['series'][0], linewidth=2 * PX, solid_joinstyle='round', zorder=3)
    ax.set_yscale('log')
    ax.axhline(100, color=t['muted'], linewidth=1 * PX, zorder=2)
    ax.annotate('gradient clip = 100', (max(x), 100), xytext=(0, 4), textcoords='offset points',
                ha='right', va='bottom', fontsize=8, color=t['ink2'])
    ax.set_xlim(0, max(x))
    ax.set_xlabel('World-model updates', color=t['ink2'], fontsize=8.5)
    ax.set_ylabel('Global gradient norm (log)', color=t['ink2'], fontsize=8.5)
    titles(fig, t, 'World-model gradient norm before clipping, CT-WM 1-hour run',
           'One logged update every ~30 s. Values above the line are scaled down to 100 before '
           'the optimizer step.')
    save(fig, 'ctwm_1h_model_gradient_norm', mode)


def ctwm_world_model(mode):
    t = THEMES[mode]
    path = HERE / 'ctwm_online_1h_world_model_evals.json'
    if not path.exists():
        return
    evals = json.loads(path.read_text())
    x = [e['counts']['model_updates'] for e in evals]
    fig, ax = figure(t)
    fig.subplots_adjust(left=0.1, right=0.84, top=0.74, bottom=0.14)
    handles, labels, ends = [], [], []
    for horizon, color in zip((1, 5, 15), t['series']):
        y = [e['metrics'][f'open_loop_{horizon}_vector_rmse'] / e['metrics'][f'persistence_{horizon}_vector_rmse']
             for e in evals]
        line, = ax.plot(x, y, color=color, linewidth=2 * PX, solid_joinstyle='round', marker='o',
                        markersize=8 * PX, markeredgecolor=t['surface'], markeredgewidth=2 * PX, zorder=3)
        handles.append(line)
        labels.append(f'{horizon}-step' if horizon > 1 else '1-step')
        ends.append((y[-1], f'{horizon}-step ×{y[-1]:.1f}'))
    ax.set_yscale('log')
    ax.axhline(1, color=t['muted'], linewidth=1 * PX, zorder=2)
    ax.annotate('= "nothing changes" baseline', (0, 1), xytext=(4, 4), textcoords='offset points',
                va='bottom', fontsize=8, color=t['ink2'])
    ticks = [0.5, 1, 2, 4, 8, 16]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f'×{v:g}' for v in ticks])
    ax.set_ylim(0.4, 16)
    ax.set_xlim(0, max(x) * 1.02 if max(x) else 1)
    end_labels(ax, t, x[-1], ends)
    ax.set_xlabel('World-model updates', color=t['ink2'], fontsize=8.5)
    legend(ax, t, handles, labels)
    titles(fig, t, 'CT-WM world model: open-loop prediction error vs. "nothing changes"',
           'RMSE of the imagined 13-vector divided by the error of repeating the last observation; '
           'below ×1 the model wins.\nSame 8 held-out episodes as validate_nrsm.py (seeds 40000-40007), '
           'every saved checkpoint of the 1-hour run.')
    save(fig, 'ctwm_1h_world_model_vs_persistence', mode)


def rounded_hbar(ax, y, width, height, color, radius_px, t):
    """Horizontal bar: 4 px rounded data end, square at the zero baseline."""
    ax.figure.canvas.draw()
    bbox = ax.get_window_extent()
    x_per_px = (ax.get_xlim()[1] - ax.get_xlim()[0]) / bbox.width
    y_per_px = (ax.get_ylim()[1] - ax.get_ylim()[0]) / bbox.height
    radius = min(radius_px * x_per_px, width / 2)
    ax.add_patch(FancyBboxPatch((0, y - height / 2), width, height, linewidth=0, facecolor=color,
                                boxstyle=f'round,pad=0,rounding_size={radius}',
                                mutation_aspect=y_per_px / x_per_px, zorder=3))
    ax.add_patch(Rectangle((0, y - height / 2), min(width, 2 * radius), height, linewidth=0,
                           facecolor=color, zorder=3))


def baselines(mode):
    t = THEMES[mode]
    runs = {r['algorithm']: r for r in json.loads(
        (HERE / 'baselines_direct_6000steps_aggregate.json').read_text())['runs']}
    order = ['pdqn', 'mpdqn', 'her_pdqn', 'her_mpdqn']
    fig, ax = figure(t, size=(7.2, 3.6))
    fig.subplots_adjust(left=0.15, right=0.95, top=0.72, bottom=0.14)
    style(ax, t, grid_axis='x')
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.invert_yaxis()
    bbox_h = ax.get_window_extent().height
    y_per_px = (len(order) - 0.2) / bbox_h
    bar = 20 * y_per_px                                         # 20 px thick (<= 24 px)
    gap = 2 * y_per_px                                          # 2 px surface gap
    for i, algorithm in enumerate(order):
        values = (100 * runs[algorithm]['success_rate'], TEAM_DIRECT_6000[algorithm])
        for j, (value, color) in enumerate(zip(values, t['series'][:2])):
            y = i + (j - 0.5) * (bar + gap)
            rounded_hbar(ax, y, value, bar, color, 4, t)
            ax.annotate(f'{value:.0f}%' if j == 0 else f'{value:.1f}%', (value, y), xytext=(5, 0),
                        textcoords='offset points', va='center', fontsize=8, color=t['ink2'])
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([NAMES[a] for a in order], color=t['ink2'], fontsize=9)
    ax.tick_params(axis='y', length=0)
    ax.set_xticks(range(0, 61, 10))
    ax.set_xticklabels([f'{v}%' for v in range(0, 61, 10)])
    ax.set_xlabel('Deterministic test success (100 episodes)', color=t['ink2'], fontsize=8.5)
    handles = [Rectangle((0, 0), 1, 1, facecolor=c, linewidth=0) for c in t['series'][:2]]
    legend(ax, t, handles, ['This Mac run (seed 0)', 'Team pilot, 2026-09-06 (mean of seeds 0-2)'])
    titles(fig, t, 'Paper baselines on the Direct task after 6,000 environment steps',
           'Same configs and budget; single-seed runs are not bit-identical across machines.')
    save(fig, 'baselines_direct_6000_steps', mode)


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    for mode in THEMES:
        ctwm_evaluation(mode)
        ctwm_gradient(mode)
        ctwm_world_model(mode)
        baselines(mode)
    print('figures written to', HERE)
