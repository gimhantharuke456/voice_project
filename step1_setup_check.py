import sys
import os

REQUIRED_PACKAGES = [
    ("torch", "torch"),
    ("torchaudio", "torchaudio"),
    ("librosa", "librosa"),
    ("soundfile", "soundfile"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("pesq", "pesq"),
    ("pystoi", "pystoi"),
    ("flask", "flask"),
]

REQUIRED_DIRS = [
    "data/clean",
    "data/noise/engine",
    "data/noise/cockpit",
    "data/noise/radio",
    "data/generated/noisy",
    "data/generated/clean",
    "models",
    "outputs/enhanced",
    "outputs/plots",
]

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"

passed = []
failed = []


def check(label, ok, detail=""):
    if ok:
        print(f"  {GREEN}[PASS]{RESET} {label}" + (f" — {detail}" if detail else ""))
        passed.append(label)
    else:
        print(f"  {RED}[FAIL]{RESET} {label}" + (f" — {detail}" if detail else ""))
        failed.append(label)


print(f"\n{BOLD}=== Aviation Speech Enhancement — Step 1: Environment Check ==={RESET}\n")

print(f"{BOLD}Python{RESET}")
version = sys.version.split()[0]
check(f"Python {version}", True)

print(f"\n{BOLD}Required packages{RESET}")
for display_name, import_name in REQUIRED_PACKAGES:
    try:
        mod = __import__(import_name)
        ver = getattr(mod, "__version__", "unknown")
        check(display_name, True, ver)
    except ImportError:
        check(display_name, False, "not installed — run: pip install " + display_name)

print(f"\n{BOLD}CUDA / GPU{RESET}")
try:
    import torch
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_name = torch.cuda.get_device_name(0)
        check("CUDA GPU", True, gpu_name)
    else:
        print(f"  {YELLOW}[INFO]{RESET} No CUDA GPU detected — training will run on CPU (use Colab for Step 6)")
except ImportError:
    print(f"  {YELLOW}[INFO]{RESET} torch not installed, skipping GPU check")

print(f"\n{BOLD}Folder structure{RESET}")
for d in REQUIRED_DIRS:
    exists = os.path.isdir(d)
    if not exists:
        os.makedirs(d, exist_ok=True)
        check(d, True, "created")
    else:
        check(d, True, "exists")

print(f"\n{'=' * 55}")
print(f"  {GREEN}{len(passed)} passed{RESET}  |  {RED}{len(failed)} failed{RESET}")
print(f"{'=' * 55}")

if failed:
    print(f"\n{RED}Fix the items above before continuing to Step 2.{RESET}")
    sys.exit(1)
else:
    print(f"\n{GREEN}All checks passed. Ready for Step 2.{RESET}\n")
