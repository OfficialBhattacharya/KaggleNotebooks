import subprocess
import sys

# Target directory for installation
TARGET_DIR = '/kaggle/working/'

# List of packages to install with pinned, compatible versions
packages = [
    "numpy==1.26.4",
    "scipy==1.11.4",
    "pandas==2.2.2",
    "scikit-learn==1.3.2",
    "matplotlib==3.8.4",
    "joblib==1.3.2",
    "pillow==10.3.0",
    "deepchem==2.8.0",
    "xgboost==2.0.3",
    "tqdm==4.66.2",
    "notebook==6.5.7",
    "seaborn==0.13.2",
    "networkx==3.2.1",
    "sympy==1.12",
    "python-dateutil==2.9.0.post0",
    "pytz==2024.1",
    "six==1.16.0",
    "threadpoolctl==3.4.0",
    "tzdata==2024.1",
    "google-auth==2.38.0",
    "google-api-core==2.10.2"
]

try:
    print(f"Installing DeepChem and compatible dependencies to {TARGET_DIR} ...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', '--upgrade', '--target', TARGET_DIR
    ] + packages)
    print("DeepChem and dependencies installation complete!")
    print(f"You can now import deepchem and other packages from {TARGET_DIR}")
except subprocess.CalledProcessError as e:
    print("Failed to install DeepChem and dependencies:", e)
    sys.exit(1) 