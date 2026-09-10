"""Check WSL driver visibility and real TensorFlow GPU execution."""

import argparse
import ctypes
import os
import pathlib
import shutil
import subprocess
import sys


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--require-gpu', action='store_true',
                      help='Exit nonzero if TensorFlow GPU execution fails.')
  args = parser.parse_args()
  print('Python:', sys.executable, flush=True)
  smi = shutil.which('nvidia-smi')
  if not smi and pathlib.Path('/usr/lib/wsl/lib/nvidia-smi').exists():
    smi = '/usr/lib/wsl/lib/nvidia-smi'
  if smi:
    try:
      result = subprocess.run([smi, '-L'], capture_output=True, text=True,
                              timeout=15, check=False)
      print('NVIDIA driver:', (result.stdout or result.stderr).strip())
    except (OSError, subprocess.TimeoutExpired) as error:
      print('NVIDIA driver check failed:', error)
  else:
    print('NVIDIA driver: nvidia-smi not found')

  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
  try:
    import tensorflow as tf
    print('TensorFlow:', tf.__version__)
    print('CUDA build:', tf.sysconfig.get_build_info())
    devices = tf.config.list_physical_devices('GPU')
    print('TensorFlow GPU devices:', devices)
    if not devices:
      raise RuntimeError('TensorFlow did not register a GPU')
    for device in devices:
      tf.config.experimental.set_memory_growth(device, True)
    # Disable CPU fallback so a successful test really proves GPU execution.
    tf.config.set_soft_device_placement(False)
    with tf.device('/GPU:0'):
      product = tf.matmul(tf.ones((32, 32)), tf.ones((32, 32)))
      convolution = tf.nn.conv2d(
          tf.ones((1, 8, 8, 1)), tf.ones((3, 3, 1, 2)),
          strides=1, padding='VALID')
      tf.debugging.assert_near(product, tf.fill((32, 32), 32.0))
      tf.debugging.assert_near(convolution, tf.fill((1, 6, 6, 2), 9.0))
      product.numpy()
      convolution.numpy()
    print('GPU_CHECK_OK: matrix multiplication and convolution on', product.device)
    return 0
  except Exception as error:
    print('GPU_CHECK_FAILED:', type(error).__name__, str(error))
    for library in ('libcuda.so.1', 'libcudart.so.12', 'libcublas.so.12',
                    'libcudnn.so.9', 'libcusolver.so.11', 'libcusparse.so.12'):
      try:
        ctypes.CDLL(library)
        print('Library OK:', library)
      except OSError as load_error:
        print('Library missing/unloadable:', library, str(load_error))
    print('LD_LIBRARY_PATH:', os.environ.get('LD_LIBRARY_PATH', '(unset)'))
    print('Use bash run_wsl.sh --check-gpu to expose the venv CUDA libraries.')
    print('Check this venv against requirements.txt if a library is missing.')
    if not args.require_gpu:
      print('CPU fallback allowed. Set CTM_REQUIRE_GPU=1 to stop before training.')
    return 1 if args.require_gpu else 0


if __name__ == '__main__':
  sys.exit(main())
