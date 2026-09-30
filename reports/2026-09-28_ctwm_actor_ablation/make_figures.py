"""Comparison figures (light + dark PNGs) for all CT-WM ablation conditions, one panel per seed.

Reads the baseline's small files from ../2026-09-28_first_run_mac/ and the ablation runs' from this folder;
runs whose files do not exist yet are skipped. Usage (any Python with numpy + matplotlib):
    python make_figures.py
Colour follows the condition (validated palette, slots 1-8): baseline, actor fix, actor fix + learned reward
std, actor fix + motion head, actor fix + motion head + normalised reward, actor fix + motion head + two-hot
reward, the same + edge head, the same + DreamerV2's demonstrations and behaviour cloning. (The one-seed 32 x 32
pilot is left out; the palette has 8 validated slots, and its numbers are in the README tables.) Lines carry direct
end labels where they do not collide; the legend and the README tables carry the rest.
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent / '2026-09-28_first_run_mac'
DPI, PX = 144, 72 / 144
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9',
                  axis='#c3c2b7', series=['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']),
    'dark': dict(surface='#1a1a19', ink='#ffffff', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a',
                 axis='#383835', series=['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767']),
}
CONDITIONS = ['Baseline (scale 0.1)', 'Actor fix (scale 1.0)', 'Actor fix + learned reward std',
              'Actor fix + motion head', 'Actor fix + motion head + normalised reward',
              'Actor fix + motion head + two-hot reward', 'Actor fix + motion head + two-hot + edge head',
              'Edge-head setup + demonstrations + BC']
SHORT = ['Baseline', 'Actor fix', '+ reward std', '+ motion head', '+ reward norm', '+ two-hot', '+ edge head',
         '+ demos + BC']
RUNS = [  # (condition index, seed, folder, file prefix)
    (0, 17, BASE, 'ctwm_online_1h'),
    (1, 17, HERE, 'imagscale1_seed17'), (1, 18, HERE, 'imagscale1_seed18'),
    (2, 17, HERE, 'imagscale1_rewardstd_seed17'), (2, 18, HERE, 'imagscale1_rewardstd_seed18'),
    (3, 17, HERE, 'imagscale1_delta_seed17'), (3, 18, HERE, 'imagscale1_delta_seed18'),
    (4, 17, HERE, 'imagscale1_delta_rewardnorm_seed17'), (4, 18, HERE, 'imagscale1_delta_rewardnorm_seed18'),
    (5, 17, HERE, 'imagscale1_delta_twohot_seed17'), (5, 18, HERE, 'imagscale1_delta_twohot_seed18'),
    (6, 17, HERE, 'imagscale1_delta_twohot_edge_seed17'), (6, 18, HERE, 'imagscale1_delta_twohot_edge_seed18'),
    (7, 17, HERE, 'imagscale01_delta_twohot_edge_demo_seed17'), (7, 18, HERE, 'imagscale01_delta_twohot_edge_demo_seed18'),
]
SEEDS = (17, 18)
WINDOW = 100
# Focus figures, experiment 8 against its imitation-only control (experiment 9). Experiment 8 keeps its colour
# from the main figures; the pair (red, blue) validates all-pairs in both modes.
FOCUS = dict(series=dict(light=['#e34948', '#2a78d6'], dark=['#e66767', '#3987e5']),
             labels=['Imitation + reinforcement learning (exp 8)', 'Imitation only (exp 9)'],
             short=['exp 8', 'imitation only'],
             runs=[(0, 17, 'imagscale01_delta_twohot_edge_demo_seed17'), (0, 18, 'imagscale01_delta_twohot_edge_demo_seed18'),
                   (1, 17, 'imagscale0_delta_twohot_edge_demo_seed17'), (1, 18, 'imagscale0_delta_twohot_edge_demo_seed18')])


def rows(folder, prefix, name):
    with (folder / f'{prefix}_{name}.csv').open() as stream:
        return list(csv.DictReader(stream))


def runs(needed):
    return [r for r in RUNS if (r[2] / f'{r[3]}_{needed}').exists()]


def rolling(values, window=WINDOW):
    return np.convolve(np.asarray(values, float), np.ones(window) / window, mode='valid')


def style(ax, t):
    ax.set_facecolor(t['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(t['axis'])
        ax.spines[side].set_linewidth(1 * PX)
    ax.tick_params(colors=t['muted'], labelcolor=t['muted'], width=1 * PX, length=3)
    ax.grid(axis='y', color=t['grid'], linewidth=1 * PX, linestyle='-')
    ax.set_axisbelow(True)


def panels(t, title, subtitle, xlabel, ylabel, log=False, top=0.63):
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.3), dpi=DPI, sharey=True)
    fig.patch.set_facecolor(t['surface'])
    fig.subplots_adjust(left=0.1, right=0.98, top=top, bottom=0.13, wspace=0.08)
    fig.text(0.012, 0.965, title, color=t['ink'], fontsize=11.5, fontweight='bold', va='top')
    fig.text(0.012, 0.905, subtitle, color=t['ink2'], fontsize=8.5, va='top')
    for ax, seed in zip(axes, SEEDS):
        style(ax, t)
        if log:
            ax.set_yscale('log')
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_title(f'Seed {seed}', color=t['ink2'], fontsize=9, loc='left', pad=4)
        ax.set_xlabel(xlabel, color=t['ink2'], fontsize=8.5)
    axes[0].set_ylabel(ylabel, color=t['ink2'], fontsize=8.5)
    return fig, dict(zip(SEEDS, axes))


def draw(ax, t, x, y, condition, ends):
    ax.plot(x, y, color=t['series'][condition], linewidth=2 * PX, solid_joinstyle='round',
            solid_capstyle='round', zorder=3)
    ends.setdefault(ax, []).append((x[-1], y[-1], condition))


def x_room(ax, data_max, tick_step):
    """Leave room inside the panel, right of the data, for end labels; ticks stop at the data."""
    ax.set_xlim(0, data_max * 1.32)
    ax.set_xticks(np.arange(0, data_max + 1e-9, tick_step))


def label_ends(fig, t, ends, short=SHORT, min_px=11):
    fig.canvas.draw()
    for ax, points in ends.items():
        placed = []
        for x, y, condition in sorted(points, key=lambda p: p[1]):
            y_px = ax.transData.transform((x, y))[1]
            if all(abs(y_px - p) >= min_px for p in placed):
                ax.annotate(short[condition], (x, y), xytext=(4, 0), textcoords='offset points',
                            va='center', fontsize=7.5, color=t['ink2'], annotation_clip=False)
                placed.append(y_px)


def finish(fig, t, present, ends, name, mode, labels=CONDITIONS, short=SHORT, ncol=3):
    label_ends(fig, t, ends, short)
    handles = [plt.Line2D([], [], color=t['series'][c], linewidth=2 * PX) for c in present]
    fig.legend(handles, [labels[c] for c in present], loc='upper left', bbox_to_anchor=(0.1, 0.84),
               ncol=ncol, frameon=False, fontsize=7.5, labelcolor=t['ink2'], handlelength=1.6,
               columnspacing=1.6)
    fig.savefig(HERE / f'{name}_{mode}.png', dpi=DPI, facecolor=fig.get_facecolor())
    plt.close(fig)


def policy(folder, prefix):
    episodes, stats = rows(folder, prefix, 'episodes'), rows(folder, prefix, 'action_stats')
    assert len(episodes) == len(stats)
    return [(e, s) for e, s in zip(episodes, stats) if e['phase'] == 'policy']


def training_curve(mode, value, name, title, ylabel, percent=True, subtitle_extra=''):
    t = THEMES[mode]
    fig, axes = panels(t, title, f'Rolling mean over {WINDOW} training episodes (sampled actions). '
                       f'Every run: 137,680 environment steps.{subtitle_extra}',
                       'Environment steps (thousands)', ylabel)
    present, ends = set(), {}
    for condition, seed, folder, prefix in runs('action_stats.csv'):
        data = policy(folder, prefix)
        x = np.array([int(e['env_steps']) for e, _ in data])[WINDOW - 1:] / 1000
        y = rolling([value(e, s) for e, s in data]) * (100 if percent else 1)
        draw(axes[seed], t, x, y, condition, ends)
        present.add(condition)
    for ax in axes.values():
        x_room(ax, 140, 20)
        ax.set_ylim(bottom=0)
        if percent:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}%'))
    finish(fig, t, sorted(present), ends, name, mode)


def focus_curve(mode, value, name, title, ylabel):
    """Experiment 8 vs its imitation-only control (experiment 9), one panel per seed."""
    t = dict(THEMES[mode], series=FOCUS['series'][mode])
    runs = [r for r in FOCUS['runs'] if (HERE / f'{r[2]}_action_stats.csv').exists()]
    if not runs:
        return
    fig, axes = panels(t, title, f'Rolling mean over {WINDOW} training episodes (sampled actions). '
                       'Every run: 137,680 environment steps.', 'Environment steps (thousands)', ylabel, top=0.72)
    present, ends = set(), {}
    for condition, seed, prefix in runs:
        data = policy(HERE, prefix)
        x = np.array([int(e['env_steps']) for e, _ in data])[WINDOW - 1:] / 1000
        draw(axes[seed], t, x, rolling([value(e, s) for e, s in data]) * 100, condition, ends)
        present.add(condition)
    for ax in axes.values():
        x_room(ax, 140, 20)
        ax.set_ylim(bottom=0)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:g}%'))
    finish(fig, t, sorted(present), ends, name, mode, FOCUS['labels'], FOCUS['short'], ncol=2)


def actor_gradient(mode):
    t = THEMES[mode]
    fig, axes = panels(t, 'Actor gradient norm during training (log scale)',
                       'One logged update every ~30 s; never clipped (clip = 100).',
                       'Actor-critic updates (thousands)', 'Gradient norm', log=True)
    present, ends = set(), {}
    for condition, seed, folder, prefix in runs('train_log.csv'):
        log = [r for r in rows(folder, prefix, 'train_log') if r['actor_gradient_norm'] and int(r['ac_updates']) > 0]
        draw(axes[seed], t, np.array([int(r['ac_updates']) / 1000 for r in log]),
             np.array([float(r['actor_gradient_norm']) for r in log]), condition, ends)
        present.add(condition)
    for ax in axes.values():
        x_room(ax, 27.5, 5)
    finish(fig, t, sorted(present), ends, 'actor_gradient_norm', mode)


def world_model(mode):
    t = THEMES[mode]
    fig, axes = panels(t, 'World model: 15-step open-loop error vs. "nothing changes" (log scale)',
                       'RMSE of the imagined 13-vector divided by the persistence error; below ×1 the model '
                       'wins. Same 8 held-out episodes.', 'World-model updates (thousands)', 'Error ratio',
                       log=True)
    present, ends = set(), {}
    for condition, seed, folder, prefix in runs('world_model_evals.json'):
        evals = json.loads((folder / f'{prefix}_world_model_evals.json').read_text())
        draw(axes[seed], t, np.array([e['counts']['model_updates'] / 1000 for e in evals]),
             np.array([e['metrics']['open_loop_15_vector_rmse'] / e['metrics']['persistence_15_vector_rmse']
                       for e in evals]), condition, ends)
        present.add(condition)
    for ax in axes.values():
        ax.axhline(1, color=t['muted'], linewidth=1 * PX, zorder=2)
        ax.set_yticks([0.5, 1, 2, 4])
        ax.set_yticklabels(['×0.5', '×1', '×2', '×4'])
        ax.set_ylim(0.4, 4)
        x_room(ax, 27.5, 5)
    finish(fig, t, sorted(present), ends, 'world_model_15step', mode)


if __name__ == '__main__':
    plt.rcParams['font.family'] = ['Helvetica Neue', 'Helvetica', 'Arial', 'DejaVu Sans']
    for mode in THEMES:
        training_curve(mode, lambda e, s: e['pickup'] == 'True', 'training_pickup_rate',
                       'Pickup rate in training episodes', 'Episodes with a pickup')
        training_curve(mode, lambda e, s: e['success'] == 'True', 'training_delivery_rate',
                       'Delivery rate in training episodes', 'Episodes delivered')
        training_curve(mode, lambda e, s: e['oob'] == 'True', 'training_oob_rate',
                       'Out-of-bounds rate in training episodes', 'Episodes ending off the map')
        training_curve(mode, lambda e, s: float(s['final_distance_to_active_goal_m']),
                       'final_distance_to_goal', 'Distance to the active goal when a training episode ends',
                       'Metres', percent=False, subtitle_extra=' Lower is better.')
        actor_gradient(mode)
        world_model(mode)
        focus_curve(mode, lambda e, s: e['pickup'] == 'True', 'imitation_control_pickup_rate',
                    'Pickups: imitation alone vs. imitation + reinforcement learning', 'Episodes with a pickup')
        focus_curve(mode, lambda e, s: e['success'] == 'True', 'imitation_control_delivery_rate',
                    'Deliveries: imitation alone vs. imitation + reinforcement learning', 'Episodes delivered')
    print('figures written to', HERE)
