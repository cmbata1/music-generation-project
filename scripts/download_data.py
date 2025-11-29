# download_data.py
import os
import gdown

# Google Drive share link (replace with your actual link)
url = "https://drive.google.com/uc?id=YOUR_FILE_ID"
output_dir = "data/processed"
output_file = os.path.join(output_dir, "full_sequences.npz")

# Make sure the folder exists
os.makedirs(output_dir, exist_ok=True)

print(f"Downloading dataset to {output_file}...")
gdown.download(url, output_file, quiet=False)
print("Download complete.")
