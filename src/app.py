import io
from pathlib import Path

import numpy as np
import torch
import streamlit as st
import requests

from model import load_trained_model
from utils import (
    build_id_to_token,
    build_token_to_id,
)

from tabs.dataset_tab import render_dataset_tab
from tabs.piano_tab import render_piano_tab
from tabs.preset_tab import render_preset_tab

# -------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent 
BASE_DIR = APP_DIR.parent                         

CHECKPOINT_PATH = BASE_DIR / "models" / "music_gru_checkpoint.pt"
FULL_SEQS_PATH = BASE_DIR / "data" / "processed" / "full_sequences.npz"
GENERATED_DIR = BASE_DIR / "generated"
GENERATED_DIR.mkdir(exist_ok=True)

DATA_URL = "https://musicgenapp.blob.core.windows.net/datasets/full_sequences.npz"

# -------------------------------------------------------------------
# Data loading with caching
# -------------------------------------------------------------------

@st.cache_resource
def load_sequences():
    """Load full_sequences.npz (local-first, blob fallback)."""
    
    # 1) Local fast path (dev or previously saved)
    if FULL_SEQS_PATH.exists():
        return np.load(FULL_SEQS_PATH, allow_pickle=True)

    # 2) Blob fallback with UI
    with st.status(
        "Downloading dataset from cloud storage…", 
        expanded=True
    ) as status:
        resp = requests.get(DATA_URL, stream=True)
        resp.raise_for_status()

        file_bytes = io.BytesIO(resp.content)
        npz = np.load(file_bytes, allow_pickle=True)

        status.update(
            label="Dataset ready ✓",
            state="complete",
            expanded=False
        )

    return npz

# -------------------------------------------------------------------
# Device selection
# -------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    device_label = "Apple Silicon GPU (MPS)"
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    device_label = "CUDA GPU"
else:
    DEVICE = torch.device("cpu")
    device_label = "CPU"

# -------------------------------------------------------------------
# Streamlit config
# -------------------------------------------------------------------
st.set_page_config(page_title="Music Generation Demo", layout="wide")

st.title("🎵 GRU Music Generator")
st.caption(f"Running on: **{device_label}**")
st.write(
    "This app showcases a GRU-based music generation model. "
    "You can generate music by selecting a preset melody seed, building a custom seed using a piano-style interface, "
    "or using an existing sequence from the dataset."
)

# -------------------------------------------------------------------
# Cache: load model + data once
# -------------------------------------------------------------------
@st.cache_resource
def load_model_and_data():
    # Load model + seq_len from checkpoint
    model, seq_len = load_trained_model(CHECKPOINT_PATH, DEVICE)

    # Load processed data
    data = np.load(FULL_SEQS_PATH, allow_pickle=True)
    X_ids = data["X"]
    vocab = data["vocab"]  # shape: (vocab_size, 2) [pitch, bucket]

    id_to_token = build_id_to_token(vocab)
    token_to_id = build_token_to_id(id_to_token)

    return model, seq_len, X_ids, id_to_token, token_to_id


model, SEQ_LEN, X_ids, id_to_token, token_to_id = load_model_and_data()

# -------------------------------------------------------------------
# Sidebar: global generation settings (shared by tabs)
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Generation Settings")

    num_tokens = st.slider(
        "Number of tokens to generate",
        min_value=50,
        max_value=500,
        value=200,
        step=50,
    )

    temps = st.multiselect(
        "Temperatures to compare",
        options=[0.4, 0.6, 0.8, 1.0, 1.2, 1.4],
        default=[0.8, 1.0],
        help="Higher temperature → more randomness; lower → more conservative.",
    )

    base_name = st.text_input(
        "Base name for output files",
        value="generated_sample",
    )

# Make sure custom seed state exists for piano tab
if "custom_seed_notes" not in st.session_state:
    st.session_state.custom_seed_notes = []  # list of (pitch, bucket)

# -------------------------------------------------------------------
# Tabs via a horizontal radio
# -------------------------------------------------------------------
mode = st.radio(
    "Select view",
    ["Preset melodies", "Custom seed (piano)", "Seed from dataset"],
    horizontal=True,
)

if mode == "Seed from dataset":
    render_dataset_tab(
        model=model,
        device=DEVICE,
        X_ids=X_ids,
        seq_len=SEQ_LEN,
        id_to_token=id_to_token,
        num_tokens=num_tokens,
        temps=temps,
        base_name=base_name,
        generated_dir=GENERATED_DIR,
    )

elif mode == "Custom seed (piano)":
    render_piano_tab(
        model=model,
        device=DEVICE,
        seq_len=SEQ_LEN,
        id_to_token=id_to_token,
        token_to_id=token_to_id,
        num_tokens=num_tokens,
        temps=temps,
        base_name=base_name,
        generated_dir=GENERATED_DIR,
    )

else:  
    render_preset_tab(
        model=model,
        device=DEVICE,
        seq_len=SEQ_LEN,
        id_to_token=id_to_token,
        token_to_id=token_to_id,
        num_tokens=num_tokens,
        temps=temps,
        base_name=base_name,
        generated_dir=GENERATED_DIR,
    )


