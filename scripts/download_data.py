import os
import gdown

url = "https://drive.google.com/uc?export=download&id=1429pDRYHCXQVSgCHpqJdEzHMUDTn5PtU"
output_dir = "data/processed"
output_file = os.path.join(output_dir, "full_sequences.npz")

os.makedirs(output_dir, exist_ok=True)

print(f"Downloading dataset to {output_file}...")
gdown.download(url, output_file, quiet=False)
print("Download complete.")
