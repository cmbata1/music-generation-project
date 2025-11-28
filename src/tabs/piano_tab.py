from pathlib import Path

import numpy as np
import streamlit as st

from utils import (
    generate_tokens,
    render_generated_sample,
    single_note_to_wav_bytes,
    seed_to_wav_bytes,
)

def render_piano_tab(
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
    Render the 'Custom seed (piano)' tab.
    Uses st.session_state.custom_seed_notes as list of (pitch, bucket).
    """
    st.subheader("Custom seed (simple piano: C4–B4)")

    # Ensure session state exists
    if "custom_seed_notes" not in st.session_state:
        st.session_state.custom_seed_notes = []

    # Duration bucket selection
    bucket_labels = {
        "Very short (0.25s)": 0,
        "Short (0.5s)": 1,
        "Medium (1.0s)": 2,
        "Long (2.0s)": 3,
    }
    bucket_label = st.selectbox(
        "Duration for custom notes",
        list(bucket_labels.keys()),
        index=1,
        key="piano_bucket_select",
    )
    chosen_bucket = bucket_labels[bucket_label]

    st.write("Click a key to add a note to your custom seed:")

    white_keys = [
        ("C4", 60),
        ("D4", 62),
        ("E4", 64),
        ("F4", 65),
        ("G4", 67),
        ("A4", 69),
        ("B4", 71),
    ]

    cols = st.columns(len(white_keys))

    # Key clicks: add note + play single note
    for col, (label, pitch) in zip(cols, white_keys):
        if col.button(label, key=f"piano_key_{label}"):
            st.session_state.custom_seed_notes.append((pitch, chosen_bucket))
            # Play the note immediately
            wav_bytes = single_note_to_wav_bytes(pitch, duration=0.5)
            st.audio(wav_bytes, format="audio/wav")

    # Seed display (collapsed by default to reduce clutter)
    with st.expander("Show current custom seed (click to expand)", expanded=False):
        if st.session_state.custom_seed_notes:
            st.write(st.session_state.custom_seed_notes)
        else:
            st.info("No notes in the seed yet.")

    # Controls: clear / play / generate
    c1, c2, c3 = st.columns(3)
    clear_clicked = c1.button("Clear seed", key="piano_clear_seed")
    play_clicked = c2.button("Play seed", key="piano_play_seed")
    generate_clicked = c3.button("Generate from seed", key="piano_generate_seed")

    # Handle clear
    if clear_clicked:
        st.session_state.custom_seed_notes = []
        # Optional: immediately stop here
        return

    # Handle "Play seed"
    if play_clicked:
        if not st.session_state.custom_seed_notes:
            st.warning("No notes to play. Add some notes first.")
        else:
            wav_bytes = seed_to_wav_bytes(st.session_state.custom_seed_notes)
            st.audio(wav_bytes, format="audio/wav")

    # Handle generation
    if not generate_clicked:
        return

    if not st.session_state.custom_seed_notes:
        st.warning("No notes in the seed. Add some notes before generating.")
        return

    if not temps:
        st.warning("Pick at least one temperature in the sidebar.")
        return

    # Convert (pitch, bucket) -> token IDs
    seed_token_ids = []
    for pitch, bucket in st.session_state.custom_seed_notes:
        key = (int(pitch), int(bucket))
        if key in token_to_id:
            seed_token_ids.append(token_to_id[key])
        else:
            st.error(f"Note {key} is not in the vocabulary. Try a different pitch/duration.")
            return

    seed_token_ids = np.array(seed_token_ids, dtype=np.int64)

    # Pad or trim to seq_len
    if len(seed_token_ids) <= seq_len:
        full_seed = seed_token_ids
    else:
        full_seed = seed_token_ids[-seq_len:]


    st.write("Using custom seed of length:", len(seed_token_ids))

    cols_custom = st.columns(len(temps))
    for temp, col in zip(temps, cols_custom):
        with col:
            st.markdown(f"**T = {temp:.2f} (custom seed)**")

            gen_ids = generate_tokens(
                model,
                full_seed,
                seq_len=seq_len,
                num_tokens=num_tokens,
                temperature=temp,
                device=device,
            )

            midi_name = f"{base_name}_piano_T{temp:.2f}.mid"
            render_generated_sample(
                gen_ids=gen_ids,
                id_to_token=id_to_token,
                generated_dir=generated_dir,
                midi_name=midi_name,
                download_key=f"piano_download_{temp}",
            )

