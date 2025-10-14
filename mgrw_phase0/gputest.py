import torch

print("🔍 Checking PyTorch GPU availability...\n")

# 显示 PyTorch 版本
print(f"PyTorch version: {torch.__version__}")

# 检查 CUDA 是否可用
cuda_available = torch.cuda.is_available()
print(f"CUDA available: {cuda_available}")

# 检查 MPS (Apple Silicon) 是否可用
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"MPS available (for Mac M1/M2): {mps_available}")

if cuda_available:
    print(f"✅ GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
elif mps_available:
    print("✅ MPS backend available on Apple Silicon (M1/M2).")
else:
    print("❌ No GPU backend detected — running on CPU only.")
