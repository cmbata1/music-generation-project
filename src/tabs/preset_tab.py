# NOTE: This file contains AI-generated code (ChatGPT).

from pathlib import Path

import numpy as np
import streamlit as st

from utils import (
    generate_tokens,
    render_generated_sample,
    seed_to_wav_bytes,
)

# AI-generated via ChatGPT
def render_preset_tab(
    model,
    device,
    seq_len: int,
    id_to_token: dict,
    token_to_id: dict,
    num_tokens: int,
    temps,
    base_name: str,
    generated_dir: Path,
):
    """
    Render the 'Preset melodies' tab.
    Seeds the model with a known melody (e.g., Twinkle Twinkle).
    """

    st.subheader("Preset melodies")

    # --- Define presets as sequences of MIDI pitches (no buckets yet) ---

    PRESETS = {
        "Twinkle Twinkle Little Star (opening)": [
            60, 60, 67, 67, 69, 69, 67,           # C C G G A A G
            65, 65, 64, 64, 62, 62, 60           # F F E E D D C
        ],
        "Frère Jacques / Are You Sleeping (opening)": [
            60, 62, 64, 60,                      # C D E C
            60, 62, 64, 60,                      # C D E C
            64, 65, 67,                          # E F G
            64, 65, 67                           # E F G
        ],
        "C major scale (C4 → C5)": [
            60, 62, 64, 65, 67, 69, 71, 72
        ]
    }

    preset_names = list(PRESETS.keys())

    preset_choice = st.selectbox(
        "Choose a preset melody",
        preset_names,
        index=0,
        key="preset_melody_select",
    )

    # Duration bucket for this preset
    bucket_labels = {
        "Very short (0.25s)": 0,
        "Short (0.5s)": 1,
        "Medium (1.0s)": 2,
        "Long (2.0s)": 3,
    }
    bucket_label = st.selectbox(
        "Duration for this melody",
        list(bucket_labels.keys()),
        index=1,
        key="preset_bucket_select",
    )
    chosen_bucket = bucket_labels[bucket_label]

    # Build (pitch, bucket) seed_notes from chosen preset
    preset_pitches = PRESETS[preset_choice]
    seed_notes = [(pitch, chosen_bucket) for pitch in preset_pitches]

    st.markdown(f"**Selected preset:** {preset_choice}")

    # Optionally let the user inspect the raw seed
    with st.expander("Show preset seed (pitch, bucket)", expanded=False):
        st.write(seed_notes)

    c1, c2 = st.columns(2)
    play_clicked = c1.button("Play preset", key="preset_play")
    generate_clicked = c2.button("Generate continuation", key="preset_generate")

    # --- Play just the preset seed ---
    if play_clicked:
        wav_bytes = seed_to_wav_bytes(seed_notes)
        st.audio(wav_bytes, format="audio/wav")

    # --- Generate continuation from the preset seed ---
    if not generate_clicked:
        return

    if not temps:
        st.warning("Pick at least one temperature in the sidebar.")
        return

    # Convert (pitch, bucket) -> token IDs
    seed_token_ids = []
    for pitch, bucket in seed_notes:
        key = (int(pitch), int(bucket))
        if key in token_to_id:
            seed_token_ids.append(token_to_id[key])
        else:
            st.error(
                f"Note {key} not in vocabulary. "
                "Try a different duration bucket for this preset."
            )
            return

    seed_token_ids = np.array(seed_token_ids, dtype=np.int64)

    if len(seed_token_ids) <= seq_len:
        # no padding — just use the seed as-is
        full_seed = seed_token_ids
    else:
        # too long → keep only the most recent seq_len tokens
        full_seed = seed_token_ids[-seq_len:]

    st.write("Using preset seed of length:", len(seed_token_ids))

    cols_custom = st.columns(len(temps))
    for temp, col in zip(temps, cols_custom):
        with col:
            st.markdown(f"**T = {temp:.2f} (preset)**")

            gen_ids = generate_tokens(
                model,
                full_seed,
                seq_len=seq_len,
                num_tokens=num_tokens,
                temperature=temp,
                device=device,
            )

            midi_name = f"{base_name}_preset_T{temp:.2f}.mid"
            render_generated_sample(
                gen_ids=gen_ids,
                id_to_token=id_to_token,
                generated_dir=generated_dir,
                midi_name=midi_name,
                download_key=f"preset_download_{temp}",
            )

