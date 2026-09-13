"""Bounded synthetic full-capacity replay test; never uses real training data."""
import argparse
import io
import json
import pathlib
import resource
import tempfile
import time

import numpy as np

import tools


def run(capacity=2_000_000, replacement_episodes=1000, temp_root=None):
  if capacity < 100 or capacity % 100 or replacement_episodes < 1:
    raise ValueError('Use a positive multiple of 100 transitions and positive replacements')
  frames = 101
  fields = ('vector', 'action', 'reward', 'discount', 'demonstration', 'phase', 'is_success')
  episode = dict(vector=np.zeros((frames, 13), np.float32),
      action=np.tile([1, 0, .1, 0], (frames, 1)).astype(np.float32),
      reward=np.zeros(frames, np.float32), discount=np.r_[np.ones(100), 0].astype(np.float32),
      demonstration=np.zeros(frames, np.float32), phase=np.zeros(frames, np.float32),
      is_success=np.zeros(frames, np.float32), image=np.zeros((frames, 64, 64, 3), np.uint8))
  stream = io.BytesIO()
  np.savez_compressed(stream, **episode)
  contents = stream.getvalue()
  count = capacity // 100
  results = []
  with tempfile.TemporaryDirectory(prefix='ctm-replay-stress-', dir=temp_root) as directory:
    root = pathlib.Path(directory)
    pinned = root / 'demonstrations'
    pinned.mkdir()
    demo = {key: np.array(value, copy=True) for key, value in episode.items()}
    demo['demonstration'][1:] = 1
    demo['is_success'][-1] = 1
    demo['phase'][50:] = 1
    demo['vector'][:] = 7
    for i in range(3):
      np.savez_compressed(pinned / f'demo-{i}-101.npz', **demo)
    for i in range(count):
      (root / f'episode-{i:08d}-101.npz').write_bytes(contents)
      if (i + 1) % 5000 == 0:
        print(f'Created {i + 1}/{count} synthetic episodes', flush=True)
    stats = {}
    sampler = tools.load_episodes(root, 1000, length=20, capacity=capacity,
        pinned_directory=pinned, priority_fraction=.5, keys=fields, statistics=stats)
    for round_index in range(4):
      if round_index >= 2:
        for i in range(count, count + replacement_episodes):
          (root / f'episode-{i:08d}-101.npz').write_bytes(contents)
        count += replacement_episodes
      started = time.perf_counter()
      demos_seen = 0
      for _ in range(1000):
        sample = next(sampler)
        assert 'image' not in sample
        demos_seen += int(np.any(sample['demonstration'] > 0))
      assert stats['cached_transitions'] == capacity + 300, stats
      assert demos_seen > 0
      assert stats['cached_bytes'] < (capacity + 300) * 100
      result = dict(round=round_index, seconds=time.perf_counter() - started,
                    demonstrations_seen=demos_seen, **stats)
      results.append(result)
      print(json.dumps(result), flush=True)
    assert results[1]['event_index_builds'] == results[0]['event_index_builds']
    assert results[-1]['loaded_episodes'] == capacity // 100 + 3 + 2 * replacement_episodes
    assert len({row['cached_bytes'] for row in results}) == 1
    result = dict(capacity=capacity, synthetic_disk_transitions=count * 100, filesystem=str(root.parent),
        pinned_transitions=300, rounds=results,
        max_process_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        previous_image_only_cache_gib=capacity * 64 * 64 * 3 / 2**30,
        passed=True)
    sampler.close()
    return result


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--capacity', type=int, default=2_000_000)
  parser.add_argument('--replacement-episodes', type=int, default=1000)
  parser.add_argument('--output', type=pathlib.Path, required=True)
  parser.add_argument('--temp-root', type=pathlib.Path, help='Parent for owned temporary synthetic replay')
  args = parser.parse_args()
  result = run(args.capacity, args.replacement_episodes, args.temp_root)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  tools.atomic_write(args.output, lambda stream: stream.write(
      json.dumps(result, indent=2, allow_nan=False).encode('utf-8')))
  print(f"PASS: cache={result['rounds'][-1]['cached_bytes'] / 2**20:.1f} MiB; "
        f"peak RSS={result['max_process_rss_mib']:.1f} MiB", flush=True)
