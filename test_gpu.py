import os
os.environ['LD_LIBRARY_PATH'] = f"{os.environ.get('CONDA_PREFIX', '')}/lib:{os.environ.get('LD_LIBRARY_PATH', '')}"

import tensorflow as tf

print("="*60)
print("TensorFlow GPU Test")
print("="*60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"Library Path: {os.environ.get('LD_LIBRARY_PATH', 'Not set')}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs Detected: {len(gpus)}")
has_gpu = len(gpus) > 0
if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i+1}: {gpu.name}")
    print("\n✓ GPU acceleration is available!")
else:
    print("\n⚠️ No GPUs detected")

print("="*60)
