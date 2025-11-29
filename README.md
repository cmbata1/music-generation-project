# Music Generation with GRUs
A sequence-modeling project that generates short musical melodies from either preset motifs or user-defined seeds.

## What it Does
This project generates short musical melodies using a GRU‑based neural network trained on classical, monophonic MIDI data. It produces new music by continuing from an initial seed sequence, which can be chosen in three ways: from preset melodies such as the C major scale, *Are You Sleeping*, or *Twinkle Twinkle Little Star*; from random snippets drawn directly from the training dataset; or from custom notes entered through a simple seven‑key piano interface ranging from middle C to B. The model aims to create sequences that evolve smoothly and sound musically plausible in pitch and timing, and an interactive Streamlit app makes it easy to select a seed, generate music, and listen instantly.


## Quick Start
All commands below assume you are in the project root directory.
### Prerequisites
- **Python 3.9–3.12**  
  - The project was originally developed with Python 3.9, but has been tested and works with 3.9, 3.10, 3.11, and 3.12.
- **pip** for installing dependencies
- (Recommended) a **virtual environment** to keep dependencies isolated

### 1. Clone the repository
```bash
git clone https://github.com/cmbata1/music-generation-project.git
cd music-generation-project
```

### 2. Create and activate a virtual environment (recommended)
```bash
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
# or
.\venv\Scripts\activate       # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the processed dataset

The processed dataset file `data/processed/full_sequences.npz` is too large to store directly in GitHub and must be downloaded from Google Drive.

#### Option A: Script
Run the helper script to download automatically into the correct folder:
```bash
python3 scripts/download_data.py
```
#### Option B: Manual
1. Download `full_sequences.npz` from [Google Drive](https://drive.google.com/uc?export=download&id=1429pDRYHCXQVSgCHpqJdEzHMUDTn5PtU).
2. Place it in `data/processed/` (create the folder if needed).

### 5. Run the Streamlit app from the project root:
```
streamlit run src/app.py
```

For training instructions and full environment setup, see SETUP.md.

## Video Links

## Evaluation

### Model Configurations (Summary)
I trained three GRU architectures:
- **small_GRU (64, 128)** — validation accuracy ~25%
- **medium_GRU (64, 192)** — validation accuracy ~26%, selected for demos due to stable training behavior
- **bigger_GRU (128, 256)** — validation accuracy ~26%, no meaningful improvement from the medium GRU

All models used a single GRU layer, dropout=0.3, Adam optimizer (`lr=1e-3`, `weight_decay=1e-5`), and the same train/validation split.

---

### Baselines vs GRU Models
Last-token and most-frequent baselines achieve ~3–4% accuracy.  
GRU models outperform by a wide margin:

| Model       | Validation Accuracy | Test Accuracy |
|-------------|---------------------|---------------|
| small_GRU   | 0.25                | —             |
| medium_GRU  | 0.26                | 0.27          |
| bigger_GRU  | 0.26               | —             |

The medium GRU generalizes well on a held‑out test set, confirming stability beyond validation.

---

### Qualitative Results
- **Temperature sampling**:  
  - T=0.8 → smoother, locally stable pitch movements  
  - T=1.4 → more jumps, higher variability  
- Generated sequences show plausible motifs but occasional dissonance or drift.

---

<details>
<summary>Detailed Metrics & Plots</summary>

#### Hyperparameter Tuning
| Config      | embed_dim | hidden_dim |
|-------------|-----------|------------|
| small_GRU   | 64        | 128        |
| bigger_GRU  | 128       | 256        |

#### Quantitative Results
| Model       | Validation Loss | Perplexity | Accuracy |
|-------------|-----------------|------------|----------|
| small_GRU   | 2.92           | 18.49      | 0.24     |
| bigger_GRU  | 2.89            | 18.08      | 0.26     |

#### Test Set Evaluation (medium GRU)
- Test loss: 2.85  
- Test perplexity: 17.37 
- Test accuracy: 0.27  

#### Plots
![Loss Curves](images/loss-curves.png)  
![Temperature 0.8](images/t-08.png)  
![Temperature 1.4](images/t-14.png)

</details>

### Additional Analysis
See notebooks/02_training for detailed training logs and model analysis. See notebooks/03_generation.ipynb for edge‑case behavior (repeated‑note, single‑token, random seeds).

### Individual Contributions
This project was completed individually.
