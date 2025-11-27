import io
import numpy as np
import torch
import pretty_midi
import scipy.io.wavfile as wavfile
from IPython.display import Audio, display
from pathlib import Path

# ------------------------------------------------------
# Duration bucket mapping
# ------------------------------------------------------
BUCKET_TO_SECONDS = {
    0: 0.25,  # very short
    1: 0.5,   # short / medium
    2: 1.0,   # medium / long
    3: 2.0,   # long
}


# ------------------------------------------------------
# Vocab / token mapping helpers
# ------------------------------------------------------
def build_id_to_token(vocab):
    """
    Given a vocab array of shape (vocab_size, 2) containing [pitch, bucket],
    return a dict id -> (pitch, bucket).
    """
    id_to_token = {
        i: (int(vocab[i, 0]), int(vocab[i, 1]))
        for i in range(vocab.shape[0])
    }
    return id_to_token


def build_token_to_id(id_to_token):
    """
    Reverse mapping (pitch, bucket) -> id
    """
    return {tok: idx for idx, tok in id_to_token.items()}


# ------------------------------------------------------
# Generation
# ------------------------------------------------------
def generate_tokens(model, seed_seq, seq_len, num_tokens=300, temperature=1.0, device="cpu"):
    """
    Generate a sequence of token IDs by repeatedly predicting the next token.
    """
    model.eval()
    generated = list(seed_seq.astype(int))

    for _ in range(num_tokens):
        # last seq_len tokens
        input_seq = np.array(generated[-seq_len:], dtype=np.int64)
        input_tensor = torch.tensor(input_seq, dtype=torch.long,
                                    device=device).unsqueeze(0)

        with torch.no_grad():
            logits = model(input_tensor)

        logits = logits[0]

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        next_id = int(np.random.choice(len(probs), p=probs))
        generated.append(next_id)

    return generated


# ------------------------------------------------------
# MIDI conversion
# ------------------------------------------------------
def tokens_to_pretty_midi(token_ids, id_to_token, out_path, program=0):
    """
    Convert token IDs into a MIDI file and save it.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=program)
    current_time = 0.0

    for tid in token_ids:
        tid = int(tid)
        if tid not in id_to_token:
            continue

        pitch, bucket = id_to_token[tid]
        duration = BUCKET_TO_SECONDS.get(bucket, 0.5)

        note = pretty_midi.Note(
            velocity=80,
            pitch=pitch,
            start=current_time,
            end=current_time + duration
        )
        inst.notes.append(note)
        current_time += duration

    pm.instruments.append(inst)
    pm.write(out_path)
    return out_path


# ------------------------------------------------------
# Play MIDI
# ------------------------------------------------------
def play_midi(path):
    """
    Synthesize audio using pretty_midi.synthesize()
    and return an Audio widget (Jupyter only).
    """
    try:
        midi = pretty_midi.PrettyMIDI(path)
        audio = midi.synthesize()
        return Audio(audio, rate=44100)
    except Exception as e:
        print("Could not synthesize audio:", e)
        return None
    

# ------------------------------------------------------
# Convert MIDI to WAV for Browser Playback
# ------------------------------------------------------
def midi_to_wav_bytes(midi_path, fs: int = 44100) -> bytes:
    """
    Load a MIDI file, synthesize audio with pretty_midi, and return WAV bytes.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    audio = pm.synthesize(fs=fs)  # float32 in [-1, 1]

    # Convert to 16-bit PCM WAV in memory
    buf = io.BytesIO()
    wavfile.write(buf, fs, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()

# ------------------------------------------------------
# Convert Single Notes to WAV for Browser Playback
# ------------------------------------------------------

def single_note_to_wav_bytes(pitch: int, duration: float = 0.5,
                             fs: int = 44100, velocity: int = 80) -> bytes:
    """
    Synthesize a single note (pitch, duration) to WAV bytes.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    note = pretty_midi.Note(
        velocity=velocity,
        pitch=pitch,
        start=0.0,
        end=duration,
    )
    inst.notes.append(note)
    pm.instruments.append(inst)

    audio = pm.synthesize(fs=fs)
    buf = io.BytesIO()
    wavfile.write(buf, fs, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()

# ------------------------------------------------------
# Convert Seed Sequence to WAV for Browser Playback
# ------------------------------------------------------
def seed_to_wav_bytes(seed_notes, fs: int = 44100, velocity: int = 80) -> bytes:
    """
    Given a list of (pitch, bucket) pairs, synthesize the whole seed
    using BUCKET_TO_SECONDS for durations, and return WAV bytes.
    """
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0)

    current_time = 0.0
    for pitch, bucket in seed_notes:
        duration = BUCKET_TO_SECONDS.get(bucket, 0.5)
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=int(pitch),
            start=current_time,
            end=current_time + duration,
        )
        inst.notes.append(note)
        current_time += duration

    pm.instruments.append(inst)
    audio = pm.synthesize(fs=fs)

    buf = io.BytesIO()
    wavfile.write(buf, fs, (audio * 32767).astype(np.int16))
    buf.seek(0)
    return buf.read()


# ------------------------------------------------------
# Multi-temperature generation helper
# ------------------------------------------------------
def generate_and_play_for_temperatures(
    model,
    X_ids,
    id_to_token,
    seq_len,
    temps=(0.8, 1.0, 1.2),
    num_tokens=300,
    base_name="generated_sample",
    use_random_seed=True,
    seed_index=None,
    device="cpu",
):
    """
    Generate continuation at several temperatures.
    Returns a dict: {temperature: path_to_midi}
    """
    if use_random_seed:
        idx = np.random.randint(0, X_ids.shape[0])
    else:
        assert seed_index is not None
        idx = seed_index

    seed_seq = X_ids[idx]

    results = {}

    for temp in temps:
        gen_ids = generate_tokens(
            model,
            seed_seq,
            seq_len=seq_len,
            num_tokens=num_tokens,
            temperature=temp,
            device=device,
        )

        out_path = f"{base_name}_T{temp:.2f}.mid"
        tokens_to_pretty_midi(gen_ids, id_to_token, out_path)

        audio_widget = play_midi(out_path)
        if audio_widget is not None:
            display(audio_widget)

        results[temp] = out_path

    return results