from pathlib import Path
import gdown

script_dir = Path(__file__).resolve().parent

data_dir = script_dir.parent

processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

output_file = processed_dir / "full_sequences.npz"

url = "https://drive.google.com/uc?export=download&id=1429pDRYHCXQVSgCHpqJdEzHMUDTn5PtU"

print(f"Downloading dataset to {output_file}...")
gdown.download(url, str(output_file), quiet=False)
print("Download complete.")
