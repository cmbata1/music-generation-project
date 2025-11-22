import os
import numpy as np
import torch
import streamlit as st

from model import load_trained_model          
from music_utils import (                     
    build_id_to_token,
    generate_tokens,
    tokens_to_pretty_midi,
)

# ------------------ CONFIG ------------------
DEVICE = "cpu"  

CHECKPOINT_PATH = "checkpoints/music_gru.ckpt"  # TODO: update if needed
VOCAB_PATH = "data/vocab.npy"                   # TODO: update if needed
X_IDS_PATH = "data/X_ids.npy"                   # TODO: update if needed
OUTPUT_DIR = "generated"                        # where to save MIDIs

st.set_page_config(page_title="Music Generator", layout="wide")


# ------------------ LOADING ------------------
@st.cache_resource
def load_model_and_data():
    # Load trained model + seq_len from your helper
    model, seq_len = load_trained_model(CHECKPOINT_PATH, DEVICE)

    # Load vocab and build mapping
    vocab = np.load(VOCAB_PATH)
    id_to_token = build_id_to_token(vocab)

    # Load seed sequences (X_ids)
    # If you saved a dict/npz, adjust accordingly:
    #   data = np.load(X_IDS_PATH)
    #   X_ids = data["X_ids"]
    X_ids = np.load(X_IDS_PATH)

    return model, id_to_token, X_ids, seq_len


def read_file_bytes(path: str) -> bytes:
    """Read any file as raw bytes (for st.audio and download)."""
    with open(path, "rb") as f:
        return f.read()


# ------------------ UI ------------------
st.title("🎵 GRU Music Generator")
st.write(
    "Generate short musical sequences from your trained GRU model. "
    "Use the sidebar to adjust the temperature, number of tokens, and seed."
)

# Load model + data once, cached
model, id_to_token, X_ids, SEQ_LEN = load_model_and_data()

# ------------- SIDEBAR CONTROLS -------------
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
    )

    seed_mode = st.radio(
        "Seed selection",
        ["Random seed from dataset", "Pick specific index"],
    )

    seed_index = None
    if seed_mode == "Pick specific index":
        max_idx = int(X_ids.shape[0] - 1)
        seed_index = st.number_input(
            "Seed index",
            min_value=0,
            max_value=max_idx,
            value=0,
            step=1,
        )

    base_name = st.text_input(
        "Base name for output files",
        value="generated_sample",
    )

    generate_button = st.button("Generate")


# ------------- MAIN BEHAVIOR -------------
if generate_button:
    # choose seed
    if seed_mode == "Random seed from dataset":
        idx = int(np.random.randint(0, X_ids.shape[0]))
    else:
        idx = int(seed_index)

    seed_seq = X_ids[idx]
    st.write(f"Using seed sequence index: **{idx}**")

    # make sure output dir exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if len(temps) == 0:
        st.warning("Please select at least one temperature.")
    else:
        cols = st.columns(len(temps))

        for temp, col in zip(temps, cols):
            with col:
                st.subheader(f"T = {temp:.2f}")

                # generate token IDs
                gen_ids = generate_tokens(
                    model,
                    seed_seq,
                    seq_len=SEQ_LEN,
                    num_tokens=num_tokens,
                    temperature=temp,
                    device=DEVICE,
                )

                # write to MIDI
                midi_name = f"{base_name}_seed{idx}_T{temp:.2f}.mid"
                midi_path = os.path.join(OUTPUT_DIR, midi_name)
                tokens_to_pretty_midi(gen_ids, id_to_token, midi_path)

                # load raw bytes
                midi_bytes = read_file_bytes(midi_path)

                # audio player (note: browser MIDI support can vary)
                st.audio(midi_bytes, format="audio/midi")

                # download button
                st.download_button(
                    label="Download MIDI",
                    data=midi_bytes,
                    file_name=midi_name,
                    mime="audio/midi",
                )
else:
    st.info("Configure the settings in the sidebar and click **Generate**.")
