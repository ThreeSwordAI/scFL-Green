import torch

def test_gpu_pytorch():
    if torch.cuda.is_available():
        print("CUDA is available!")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Index: {torch.cuda.current_device()}")
    else:
        print("CUDA is not available. Please check your GPU setup.")

test_gpu_pytorch()