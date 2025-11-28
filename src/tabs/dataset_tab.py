# NOTE: This file contains AI-generated code (ChatGPT).

from pathlib import Path

import numpy as np
import streamlit as st

from utils import (
    generate_tokens,
    render_generated_sample,
)

# AI-generated via ChatGPT
def render_dataset_tab(
    model,
    device,
    X_ids,
    seq_len: int,
    id_to_token: dict,
    num_tokens: int,
    temps,
    base_name: str,
    generated_dir: Path,
):
    """
    Render the 'Seed from dataset' tab.
    """
    st.subheader("Use an existing sequence as the seed")

    seed_mode = st.radio(
        "Seed selection method",
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
            help=f"Must be between 0 and {max_idx}.",
        )

    generate_button = st.button("Generate from dataset seed")

    if not generate_button:
        return

    if len(temps) == 0:
        st.warning("Please select at least one temperature in the sidebar.")
        return

    # Choose seed sequence
    if seed_mode == "Random seed from dataset":
        idx = int(np.random.randint(0, X_ids.shape[0]))
    else:
        idx = int(seed_index)

    seed_seq = X_ids[idx]
    st.write(f"Using seed sequence index: **{idx}**")
    st.write(f"Sequence length (SEQ_LEN): **{seq_len}**")

    cols = st.columns(len(temps))

    for temp, col in zip(temps, cols):
        with col:
            st.markdown(f"**T = {temp:.2f}**")

            # Generate token IDs
            gen_ids = generate_tokens(
                model,
                seed_seq,
                seq_len=seq_len,
                num_tokens=num_tokens,
                temperature=temp,
                device=device,
            )

            midi_name = f"{base_name}_seed{idx}_T{temp:.2f}.mid"
            render_generated_sample(
                gen_ids=gen_ids,
                id_to_token=id_to_token,
                generated_dir=generated_dir,
                midi_name=midi_name,
                download_key=f"dataset_download_{temp}",
            )

