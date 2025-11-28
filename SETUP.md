# SETUP

This document provides installation and setup instructions for running the music generation project.

## 1. Install Dependencies

All commands assume you are in the project root directory.

Install required packages:

```
pip install -r requirements.txt
```

(Optional) If you prefer a virtual environment, you may create and activate one before installing dependencies:
```
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# or
.\venv\Scripts\activate       # Windows PowerShell
```

## 2. Download the Processed Dataset
The file `data/processed/full_sequences.npz` is too large to store directly in the GitHub repository.

Download it using the link below:

[Download full_sequences.npz from Google Drive](https://drive.google.com/uc?export=download&id=1khv6_3rRCDJ27viPqy3eqDezGD2r02KQ)

After downloading, place the file inside the `data/processed/` folder. The `data/processed/` folder should now contain:
-   `full_sequences.npz`
-   `splits_filelevel_indices.npz`

This file is **required** for the Streamlit app to run.

## 3. Running the Streamlit App
The interactive music generation application is run through Streamlit.

From the project root, run:
```
streamlit run src/app.py
```

This will launch the app in your browser. The app automatically loads:
- The trained GRU model from `models/music_gru_checkpoint.pt`
- The preprocessed dataset from `data/processed/full_sequences.npz`

## 4. Running the Notebooks (Optional)
Training and analysis notebooks are included for transparency but **do not need to be rerun** to evaluate the project.

You can view them directly on GitHub, or open them locally in your preferred notebook environment (e.g., VS Code, JupyterLab, or the classic Jupyter Notebook interface).

The main notebooks are: 
-   `notebooks/01_preprocessing.ipynb` — converts raw MIDI files into token sequences  
-   `notebooks/02_training.ipynb` — trains the GRU model  
-   `notebooks/03_generation.ipynb` — qualitative analysis and plots  

All outputs are saved so they display correctly even without re-execution.

## 5. Troubleshooting
### Streamlit not found?
Make sure dependencies were installed:
`pip install -r requirements.txt`

### Missing dataset file?
Ensure `full_sequences.npz` is downloaded and located at:
`data/processed/full_sequences.npz`. If this file is missing, the Streamlit app will not start.


### GPU Note:
The app automatically selects CPU, CUDA, or Apple MPS depending on your system.

This completes the setup. For full project details, see `README.md`.
