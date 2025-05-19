import tensorflow as tf

def test_gpu_tensorflow():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"{len(gpus)} GPU(s) available!")
        for gpu in gpus:
            print(f"Device Name: {gpu.name}")
    else:
        print("No GPU found.")

test_gpu_tensorflow()