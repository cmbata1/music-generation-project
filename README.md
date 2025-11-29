# Music Generation with GRUs
A sequence-modeling project that generates short musical melodies from either preset motifs or user-defined seeds.

## What it Does
This project generates short musical melodies using a GRU-based neural network trained on tokenized, monophonic MIDI files drawn from a classical music dataset. The system produces new music by continuing from an initial seed sequence, which can come from the dataset, from preset melodies, or from user-provided custom notes. The model aims to produce sequences where token values evolve smoothly and reflect musically plausible movement in pitch and timing. An interactive Streamlit app allows users to choose a seed type, generate music, and listen to their output.

## Quick Start
All commands below assume you are in the project root directory.
### Prerequisites
- Python 3.12 (this is the version the project was developed and tested on)
- pip for installing dependencies
- (Recommended) virtual environment

### 1. Clone the repository
```bash
git clone https://github.com/cmbata1/music-generation-project.git
cd music-generation-project
```

### 2. Create and activate a virtual environment (recommended)
```bash
python3.12 -m venv venv
source venv/bin/activate      # macOS / Linux
# or
.\venv\Scripts\activate       # Windows PowerShell
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the processed dataset
The Streamlit app cannot run without the processed sequence file.

Download [full_sequences.npz](https://drive.google.com/uc?export=download&id=1khv6_3rRCDJ27viPqy3eqDezGD2r02KQ).
    
Place it in `data/processed/` (create the folder if needed)

### 5. Run the Streamlit app from the project root:
```
streamlit run src/app.py
```

For training instructions and full environment setup, see SETUP.md.

## Video Links

## Evaluation
### Model Configurations

I trained several GRU architectures during development:

- **Medium GRU (64, 192)** — trained for 10 epochs; used to verify training stability and to produce qualitative samples.
- **small_GRU (64, 128)** — trained for 6 epochs; used for hyperparameter comparison.
- **bigger_GRU (128, 256)** — trained for 6 epochs; achieved the best validation performance and serves as the primary quantitative model.

All models use a single GRU layer with dropout of 0.3, Adam optimization (`lr=1e-3`, `weight_decay=1e-5`), and the same train/validation split.

---

### Hyperparameter Tuning

To compare model capacity, two GRU configurations were trained on the same train/validation split:

| Config      | embed_dim | hidden_dim | Val Loss | Perplexity | Val Acc |
|-------------|-----------|------------|----------|------------|---------|
| small_GRU   | 64        | 128        | 2.98     | 19.77      | 0.24    |
| bigger_GRU  | 128       | 256        | 2.93     | 18.86      | 0.26    |

The larger GRU consistently achieved lower validation loss and perplexity, as well as higher accuracy, so it serves as the primary model for quantitative evaluation.

---

### Quantitative Results
To contextualize model performance, two simple baselines were evaluated:

| Baseline Method         | Accuracy |
|-------------------------|----------|
| Majority-token baseline | 0.03     |
| Last-token baseline     | 0.03     |

Both GRU models outperform these baselines:

| Model       | Validation Loss | Perplexity | Accuracy |
|-------------|------------------|------------|----------|
| small_GRU   | 2.98             | 19.77      | 0.24     |
| bigger_GRU  | 2.93             | 18.86      | 0.26     |

The larger GRU achieves lower validation loss, lower perplexity, and higher top-1 accuracy, indicating improved next-token prediction quality.

Below is a representative training and validation loss curve from the medium-sized GRU model trained for 10 epochs. This model was used to verify training stability and to produce qualitative samples.

![Loss Curves](images/loss-curves.png)

---

### Test Set Evaluation

To assess generalization, the medium GRU model (`embed_dim=64`, `hidden_dim=192`) was evaluated on a held-out test split that was not used during training or hyperparameter tuning.

Baseline performance on the test set:

| Baseline Method         | Accuracy |
|-------------------------|----------|
| Majority-token baseline | 0.04     |
| Last-token baseline     | 0.04     |

Model performance:

- Test loss: 2.86  
- Test perplexity: 17.52  
- Test accuracy: 0.28  

These results show that the model continues to outperform naïve baselines by a large margin, even on unseen data. This complements the validation-based comparisons used during model selection and demonstrates that the GRU generalizes beyond the training distribution.

---

### Qualitative Results
#### Effect of Sampling Temperature on Musical Structure

To qualitatively inspect the model’s behavior, sequences were generated from the medium GRU model trained for 10 epochs using the same seed at two temperatures:

- **T = 0.8** (moderate diversity)  
- **T = 1.4** (high diversity)

Pitch-vs-step plots show how musical structure evolves over time.

![Temperature 0.8](images/t-08.png)
![Temperature 1.4](images/t-14.png)

Observations:
- **T = 0.8** produces smoother, more locally stable pitch movements.  
- **T = 1.4** introduces larger jumps and higher variability.  
- Lower temperatures keep sampling near high-probability predictions, while higher temperatures increase randomness and reduce coherence.

### Additional Analysis
See notebooks/03_generation.ipynb for edge‑case behavior (repeated‑note, single‑token, random seeds).

### Individual Contributions
This project was completed individually.
