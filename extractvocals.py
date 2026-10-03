"""
Extract vocal range characteristics from a folder of song audio files.

Pipeline per song:
  1. Demucs (command line) separates the vocal stem from the full mix
  2. librosa's pYIN tracks pitch (F0) on the isolated vocal
  3. The raw pitch contour is summarized into: lowest/highest note,
     tessitura (comfortable range), vibrato presence, and loudness dynamics

Output: a single CSV with one row per song.

Install dependencies first:
    pip install demucs librosa soundfile numpy certifi "torchaudio<2.9"

(The torchaudio<2.9 pin matters: newer torchaudio versions force audio
saving through a package called torchcodec, which Demucs doesn't bundle,
causing a "TorchCodec is required" crash. Pinning below 2.9 avoids that.)

Usage:
    python extract_vocal_features.py
    (a Finder/Explorer window will pop up asking you to select your songs folder,
    then it'll ask for an output filename in the terminal)
"""

import os
import csv
import subprocess
import sys
from pathlib import Path

# Fix for macOS "SSL: CERTIFICATE_VERIFY_FAILED" errors when Demucs downloads
# its pretrained model weights on first run. The python.org Python build doesn't
# trust the system's root certificates by default, so we point it at certifi's
# bundle instead. Must happen before any network calls (Demucs/torch.hub).
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass  # certifi not installed; if you hit SSL errors, run: pip install certifi

import librosa
import numpy as np

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg"}


def separate_vocals(song_path: Path, work_dir: Path) -> Path:
    """Run Demucs to isolate the vocal stem. Returns path to the vocals wav."""
    subprocess.run(
        [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",
            "-o", str(work_dir),
            str(song_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    # Demucs writes to work_dir/htdemucs/<song_stem>/vocals.wav
    vocals_path = work_dir / "htdemucs" / song_path.stem / "vocals.wav"
    return vocals_path


def extract_pitch_contour(vocals_path: Path):
    """Load the isolated vocal track and run pYIN pitch tracking."""
    y, sr = librosa.load(str(vocals_path), sr=22050, mono=True)
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz("C2"),   # ~65 Hz, low bass floor
        fmax=librosa.note_to_hz("C6"),   # ~1047 Hz, high soprano ceiling
        sr=sr,
    )
    rms = librosa.feature.rms(y=y)[0]
    # timbre brightness, same measurement used on the user-voice side so the
    # two are directly comparable in the matching engine
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = float(np.mean(spectral_centroids))
    hop_rate = sr / 512  # frames per second at default hop_length
    return f0, voiced_flag, rms, hop_rate, mean_centroid


def detect_vibrato(f0_voiced: np.ndarray, hop_rate: float) -> bool:
    """Heuristic: check for periodic pitch oscillation in the 4-7 Hz band,
    the typical human vibrato rate."""
    if len(f0_voiced) < hop_rate * 2:
        return False
    cents = 1200 * np.log2(f0_voiced / np.nanmedian(f0_voiced))
    cents = cents - np.nanmean(cents)
    cents = np.nan_to_num(cents)
    fft_vals = np.abs(np.fft.rfft(cents))
    freqs = np.fft.rfftfreq(len(cents), d=1 / hop_rate)
    band = (freqs >= 4) & (freqs <= 7)
    if not band.any():
        return False
    band_power = fft_vals[band].max()
    total_power = fft_vals[1:].max() if len(fft_vals) > 1 else 1
    return band_power > 0.25 * total_power


def summarize_song(f0, voiced_flag, rms, hop_rate, mean_centroid):
    voiced_f0 = f0[voiced_flag]
    voiced_f0 = voiced_f0[~np.isnan(voiced_f0)]
    if len(voiced_f0) < 10:
        return None  # not enough signal, skip (likely instrumental or separation failed)

    # clip extreme outliers (stray noise) before computing range
    low_clip, high_clip = np.percentile(voiced_f0, [1, 99])
    clipped = voiced_f0[(voiced_f0 >= low_clip) & (voiced_f0 <= high_clip)]

    lowest_hz, highest_hz = float(clipped.min()), float(clipped.max())
    tess_low_hz, tess_high_hz = (float(v) for v in np.percentile(clipped, [25, 75]))

    return {
        "lowest_hz": round(lowest_hz, 2),
        "highest_hz": round(highest_hz, 2),
        "tessitura_low_hz": round(tess_low_hz, 2),
        "tessitura_high_hz": round(tess_high_hz, 2),
        "lowest_note": librosa.hz_to_note(lowest_hz),
        "highest_note": librosa.hz_to_note(highest_hz),
        "tessitura_low": librosa.hz_to_note(tess_low_hz),
        "tessitura_high": librosa.hz_to_note(tess_high_hz),
        "timbre_centroid_hz": round(mean_centroid, 1),
        "vibrato_detected": detect_vibrato(clipped, hop_rate),
        "mean_loudness_rms": round(float(np.mean(rms)), 4),
        "dynamic_range_rms": round(float(np.percentile(rms, 95) - np.percentile(rms, 5)), 4),
    }


def ask_for_folder() -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)  # bring the dialog to the front
        selected = filedialog.askdirectory(title="Select your songs folder")
        root.destroy()
        if selected:
            return Path(selected)
        print("No folder selected, falling back to typing a path.")
    except Exception:
        print("Couldn't open a folder picker window, falling back to typing a path.")

    while True:
        raw = input("Folder containing your downloaded songs: ").strip().strip('"')
        folder = Path(raw).expanduser()
        if folder.is_dir():
            return folder
        print(f"  '{folder}' is not a valid folder, try again.")


def ask_for_output_name() -> Path:
    raw = input("Output CSV filename [vocal_db.csv]: ").strip()
    if not raw:
        raw = "vocal_db.csv"
    if not raw.lower().endswith(".csv"):
        raw += ".csv"
    return Path(raw)


def main():
    print("=== Vocal Feature Extractor ===\n")
    folder = ask_for_folder()
    out_path = ask_for_output_name()
    work_dir = Path("separated_tmp")

    songs = [p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTS]
    if not songs:
        print(f"No audio files found in {folder} (looked for {sorted(AUDIO_EXTS)})")
        return

    print(f"\nFound {len(songs)} song(s). Starting extraction...\n")
    rows = []

    for i, song in enumerate(songs, 1):
        print(f"[{i}/{len(songs)}] Processing {song.name} ...")
        try:
            vocals_path = separate_vocals(song, work_dir)
            f0, voiced_flag, rms, hop_rate, mean_centroid = extract_pitch_contour(vocals_path)
            summary = summarize_song(f0, voiced_flag, rms, hop_rate, mean_centroid)
            if summary is None:
                print(f"  -> skipped (insufficient vocal signal)")
                continue
            summary["filename"] = song.name
            rows.append(summary)
        except Exception as e:
            print(f"  -> failed: {e}")

    fieldnames = [
        "filename",
        "lowest_hz", "highest_hz", "tessitura_low_hz", "tessitura_high_hz",
        "lowest_note", "highest_note", "tessitura_low", "tessitura_high",
        "timbre_centroid_hz", "vibrato_detected",
        "mean_loudness_rms", "dynamic_range_rms",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. Wrote {len(rows)} songs to {out_path}")


if __name__ == "__main__":
    main()