# SETUP.md

This document provides installation and setup instructions for running the music generation project.

## 1. Install Dependencies

All commands assume you are in the project root directory.

Install required packages:

```
pip install -r requirements.txt
```

(Optional) If you prefer a virtual environment:
```
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# or
.\venv\Scripts\activate       # Windows PowerShell
```

## 2. Running the Streamlit App
The interactive music generation application is run through Streamlit.

From the project root, run:
streamlit run src/app.py

This will launch the app in your browser. The app automatically loads:
- The trained GRU model from `models/music_gru_checkpoint.pt`
- The preprocessed dataset from `data/processed/full_sequences.npz`