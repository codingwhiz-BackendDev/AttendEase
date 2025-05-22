import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# List of required directories
required_dirs = [
    'media',
    'media/faces',
    'static',
    'staticfiles',
]

print("Checking and creating required directories...")

for dir_path in required_dirs:
    full_path = os.path.join(BASE_DIR, dir_path)
    if not os.path.exists(full_path):
        os.makedirs(full_path)
        print(f"Created directory: {full_path}")
    else:
        print(f"Directory already exists: {full_path}")

print("\nDirectory setup complete!")