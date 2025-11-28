# Music Generation with GRUs
A sequence-modeling project that generates short musical melodies from either preset motifs or user-defined seeds.

## What it Does
This project generates short musical melodies using a GRU-based neural network trained on tokenized, monophonic MIDI files drawn from a classical music dataset. The system produces new music by continuing from an intial seed sequence, which can come from the dataset, from preset melodies, or from user-provided custom notes. The model aims to produce sequences where token values evolve smoothly rather than jumping abruptly, reflecting musically plausible movement in pitch and timing. An interactive interface allows users to choose a seed type, generate music, and listen to their output.

## Quick Start
All commands below assume you are in the project root directory.

1. Clone the repository
```
git clone https://github.com/cmbata1/music-generation-project.git
cd music-generation-project
```

3. Install dependencies
```
pip install -r requirements.txt
```

5. Run the Streamlist app
From the project root:
```
streamlit run src/app.py
```

For training instructions and full environment setup, see SETUP.md.
