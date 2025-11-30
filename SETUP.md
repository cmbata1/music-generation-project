# SETUP

This document provides installation and setup instructions for running the music generation project.

## Requirements

- Python 3.12 (this is the version the project was developed and tested on)
- pip for installing dependencies
- (Recommended) virtual environment (`venv` or `conda`)

## 1. Clone the repository
```bash
git clone https://github.com/cmbata1/music-generation-project.git
cd music-generation-project
```

## 2. Create and activate a virtual environment 
```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# or
.\venv\Scripts\activate       # Windows PowerShell
```

To deactivate later:
```bash
deactivate
```

## 3. Install Dependencies
With the virtual environment active:

```bash
pip install -r requirements.txt
```

To update dependencies:
```bash
pip install --upgrade -r requirements.txt
```

## 4. Download the Processed Dataset
The processed dataset file `data/processed/full_sequences.npz` is too large to store directly in GitHub and must be downloaded from Google Drive.

#### Option A: Script
Run the helper script to download automatically into the correct folder:
```bash
python scripts/download_data.py
```
#### Option B: Manual
1. Download `full_sequences.npz` from [Google Drive](https://drive.google.com/uc?export=download&id=1429pDRYHCXQVSgCHpqJdEzHMUDTn5PtU).
2. Place it in `data/processed/` (create the folder if needed).

This file is **required** for the Streamlit app.

## 5. Running the Streamlit App
The interactive music generation application is run through Streamlit.

From the project root, run:
```bash
streamlit run src/app.py
```

This will launch the app in your browser. The app loads:
- The trained GRU model from `models/music_gru_checkpoint.pt`
- The preprocessed dataset from `data/processed/full_sequences.npz`

## 6. Running the Notebooks (Optional)
Training and analysis notebooks are included for transparency but **do not need to be rerun**.

You can view them directly on GitHub, or open them locally in your preferred notebook environment (e.g., VS Code, JupyterLab).

The main notebooks are: 
-   `notebooks/01_preprocessing.ipynb` — converts raw MIDI files into token sequences  
-   `notebooks/02_training.ipynb` — trains the GRU model  
-   `notebooks/03_generation.ipynb` — qualitative analysis and plots  

Outputs are pre‑saved for viewing without re‑execution.

## 7. Troubleshooting
### Streamlit not found?
Make sure dependencies were installed:
`pip install -r requirements.txt`

### Missing dataset file?
Ensure `full_sequences.npz` is downloaded and located at:
`data/processed/full_sequences.npz`. If this file is missing, the Streamlit app will not start.

### GPU Note:
The app automatically selects CPU, CUDA, or Apple MPS depending on your system.
