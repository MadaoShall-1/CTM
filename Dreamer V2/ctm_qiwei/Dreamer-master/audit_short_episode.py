"""Reproduce historical short-replay skipping without starting training."""
import argparse
import ast
import collections
import hashlib
import json
import pathlib
import re
import subprocess
import tempfile
import time

import numpy as np


class RepeatedSkip(RuntimeError):
  pass


def git_source(repository, revision):
  return subprocess.run(['git', '-C', str(repository), 'show', revision],
                        capture_output=True, text=True, check=True).stdout


def isolated_loader(source, messages):
  """Execute only the repository's loader function, without importing TensorFlow."""
  definition = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == 'load_episodes')
  def capped_print(*values, **kwargs):
    messages.append(' '.join(map(str, values)))
    if len(messages) >= 3:
      raise RepeatedSkip('Stopped diagnostic after three repeated skips')
  namespace = dict(np=np, pathlib=pathlib, time=time, print=capped_print)
  exec(compile(ast.Module(body=[definition], type_ignores=[]), '<replay-loader>', 'exec'), namespace)
  return namespace['load_episodes']


def probe(source, stored_frames, required):
  messages = []
  loader = isolated_loader(source, messages)
  with tempfile.TemporaryDirectory(prefix='ctm-short-replay-audit-') as temporary:
    directory = pathlib.Path(temporary)
    episode = dict(observation=np.arange(stored_frames, dtype=np.float32)[:, None],
        action=np.zeros((stored_frames, 6), np.float32),
        reward=np.r_[np.zeros(stored_frames - 1), -1].astype(np.float32),
        discount=np.r_[np.ones(stored_frames - 1), 0].astype(np.float32))
    np.savez_compressed(directory / f'synthetic-single-episode-{stored_frames}.npz', **episode)
    try:
      sample = next(loader(directory, 1, length=required, seed=0))
      result = dict(yielded=True, returned_frames=len(sample['reward']),
          valid=sample.get('valid', np.ones(required)).tolist(),
          sampled_observation_indices=sample['observation'][:, 0].tolist(),
          terminal_preserved=bool(np.any((sample['reward'] == -1) & (sample['discount'] == 0))),
          returned_rewards=sample['reward'].tolist(),
          returned_discounts=sample['discount'].tolist())
    except RepeatedSkip:
      result = dict(yielded=False, repeated_skip_guard_triggered=True)
    return dict(stored_frames=stored_frames, real_transitions=stored_frames-1,
                required=required, source_episode_files=1, messages=messages, **result)


def terminal_sampling_probe(source):
  messages = []
  loader = isolated_loader(source, messages)
  with tempfile.TemporaryDirectory(prefix='ctm-terminal-sampling-audit-') as temporary:
    directory = pathlib.Path(temporary)
    np.savez_compressed(directory / 'synthetic-single-episode-11.npz',
        action=np.zeros((11, 6), np.float32), reward=np.r_[np.zeros(10), -1],
        discount=np.r_[np.ones(10), 0])
    generator = loader(directory, 1, length=10, seed=0)
    hits = sum(bool(np.any(next(generator)['discount'] == 0)) for _ in range(64))
    return dict(stored_frames=11, required=10, sampled_sequences=64, terminal_hits=hits)


def real_run_summary(directory):
  histogram = collections.Counter()
  short_files, nine_files = [], []
  for path in sorted((directory / 'episodes').glob('*.npz')):
    with np.load(path) as episode:
      length = len(episode['reward'])
      histogram[length] += 1
      if length < 20:
        row = dict(file=path.name, stored_frames=length, real_transitions=length-1,
            out_of_bounds=bool(episode['out_of_bounds'][-1]),
            success=bool(episode['is_success'][-1]))
        short_files.append(row)
        if length == 9:
          nine_files.append(row)
  log = directory / 'console.log'
  skips = collections.Counter()
  if log.exists():
    for length, required in re.findall(
        r'Skipped short episode: episode_length=(\d+), required=(\d+)',
        log.read_text(errors='replace')):
      skips[f'episode_length={length}, required={required}'] += 1
  return dict(directory=str(directory), episode_count=sum(histogram.values()),
      environment_steps=sum((length-1)*count for length, count in histogram.items()),
      stored_frame_histogram=dict(sorted(histogram.items())),
      short_under20_count=len(short_files), nine_frame_episodes=nine_files,
      total_skip_messages=sum(skips.values()), skip_message_counts=dict(skips),
      short_examples=short_files[:8])


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  args = parser.parse_args()
  project = pathlib.Path(__file__).resolve().parent
  ctm = project.parents[2]
  sources = dict(
      historical_v2_fb26c45=git_source(project.parent, 'fb26c45:Dreamer-master/tools.py'),
      current_v2=(project / 'tools.py').read_text(),
      frozen_validated_v2=(project / 'outputs/v2_corrected_validation_20260910/source/tools.py').read_text(),
      published_ctm_5d5a91a=git_source(ctm, '5d5a91a:Dreamer V2/ctm_qiwei/Dreamer-master/tools.py'),
      separate_root_dreamer=(ctm / 'Dreamer-master/tools.py').read_text())
  result = {}
  for label, source in sources.items():
    tests = [probe(source, frames, required) for frames, required in
             [(9, 10), (10, 10), (9, 20), (9, 50), (11, 10)]]
    result[label] = dict(source_sha256=hashlib.sha256(source.encode()).hexdigest(), tests=tests,
                         terminal_sampling=terminal_sampling_probe(source))
  for label in ['current_v2', 'frozen_validated_v2', 'published_ctm_5d5a91a']:
    for record in result[label]['tests']:
      assert record['yielded']
      assert sum(record['valid']) == min(record['stored_frames'], record['required'])
      if record['stored_frames'] <= record['required']:
        assert record['terminal_preserved']
    assert result[label]['tests'][0]['valid'] == [1.] * 9 + [0.]
  old = result['historical_v2_fb26c45']['tests'][0]
  assert old['messages'] == ['Skipped short episode: episode_length=9, required=10.'] * 3
  assert not old['yielded']
  assert result['separate_root_dreamer']['terminal_sampling']['terminal_hits'] == 0
  assert result['current_v2']['terminal_sampling']['terminal_hits'] > 0
  args.output.parent.mkdir(parents=True, exist_ok=True)
  args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
  print(json.dumps({label: dict(nine_frames=value['tests'][0],
      exactly_required_yielded=value['tests'][1]['yielded']) for label, value in result.items()}, indent=2))
  print('All current/frozen/published V2 short-episode assertions passed.')
  runs = {name: real_run_summary(project / 'outputs' / name)
          for name in ['v2_gpu_relay_20260910', 'v2_corrected_validation_20260910/seed_0']}
  args.output.with_name('real_runs.json').write_text(json.dumps(runs, indent=2), encoding='utf-8')
  print(json.dumps({name: {key: value for key, value in record.items()
      if key not in ['stored_frame_histogram', 'short_examples', 'skip_message_counts']}
      for name, record in runs.items()}, indent=2))


if __name__ == '__main__':
  main()
