"""Small, TensorFlow-free persistence and exclusive-run helpers."""
import json
import os
import pathlib
import shutil
import uuid


def atomic_write(path, write):
  path = pathlib.Path(path)
  path.parent.mkdir(parents=True, exist_ok=True)
  temporary = path.with_name(f'.{path.name}.{uuid.uuid4().hex}.tmp')
  try:
    with temporary.open('xb') as stream:
      write(stream)
      stream.flush()
      os.fsync(stream.fileno())
    os.replace(temporary, path)
  finally:
    temporary.unlink(missing_ok=True)


def require_free_space(directory, minimum_bytes):
  if minimum_bytes and shutil.disk_usage(directory).free < minimum_bytes:
    raise RuntimeError(f'Insufficient disk space in {directory}; stopping before writing')


def snapshot_checkpoint(checkpoint, step, minimum_bytes=0):
  """Retain a complete checkpoint without buffering another copy in memory."""
  checkpoint = pathlib.Path(checkpoint)
  if not isinstance(step, int) or step < 0:
    raise ValueError('Checkpoint step must be a nonnegative integer')
  require_free_space(checkpoint.parent, minimum_bytes + checkpoint.stat().st_size)
  destination = checkpoint.parent / 'checkpoints' / f'step_{step:09d}.pkl'
  if destination.exists():
    raise FileExistsError(f'Refusing to replace retained checkpoint: {destination}')
  with checkpoint.open('rb') as source:
    atomic_write(destination, lambda stream: shutil.copyfileobj(source, stream))
  return destination


def read_jsonl(path):
  """Stream records, ignoring only an interrupted final (unterminated) line."""
  with pathlib.Path(path).open(encoding='utf-8') as stream:
    for line in stream:
      if not line.strip():
        continue
      try:
        yield json.loads(line)
      except json.JSONDecodeError:
        if not line.endswith('\n'):
          return
        raise


def append_jsonl(path, value, durable=False):
  """Repair a torn tail before append, without rereading a growing log."""
  with pathlib.Path(path).open('a+b') as stream:
    end = stream.tell()
    if end:
      stream.seek(end - 1)
      if stream.read(1) != b'\n':
        start, tail = end, b''
        while start and b'\n' not in tail:
          size = min(4096, start)
          start -= size
          stream.seek(start)
          tail = stream.read(size) + tail
        suffix = tail.rsplit(b'\n', 1)[-1]
        try:
          json.loads(suffix)
        except (json.JSONDecodeError, UnicodeDecodeError):
          stream.truncate(end - len(suffix))
        else:
          stream.seek(0, os.SEEK_END)
          stream.write(b'\n')
    stream.seek(0, os.SEEK_END)
    stream.write((json.dumps(value, allow_nan=False, sort_keys=True) + '\n').encode('utf-8'))
    stream.flush()
    if durable:
      os.fsync(stream.fileno())


class RunLock:
  """OS-owned lock: crashes release it; a stale PID file cannot block a resume."""
  def __init__(self, path):
    self.path = pathlib.Path(path)
    self.stream = None

  def __enter__(self):
    self.path.parent.mkdir(parents=True, exist_ok=True)
    self.stream = self.path.open('a+b')
    try:
      if os.name == 'nt':
        import msvcrt
        self.stream.seek(0)
        if not self.stream.read(1):
          self.stream.write(b'0')
          self.stream.flush()
        self.stream.seek(0)
        msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
      else:
        import fcntl
        # POSIX record locks are not inherited by forked environment workers.
        fcntl.lockf(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as error:
      self.stream.close()
      self.stream = None
      raise RuntimeError(f'Another process is already using {self.path.parent}') from error
    return self

  def __exit__(self, *_):
    if self.stream is not None:
      self.stream.close()
      self.stream = None
