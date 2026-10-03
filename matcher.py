"""
Match a user's vocal profile against the song database to recommend songs
suited to their voice.

Scoring is a weighted blend of four factors:
  - range fit (45%):    can the user actually hit every note the song needs?
  - comfort fit (30%):  how close is the song's natural register to where
                         the user's voice sits comfortably?
  - timbre fit (15%):   how similar is the song's vocal brightness/color
                         to the user's?
  - style fit (10%):    do they match on vibrato usage?

All pitch comparisons are done in semitones (a musical/logarithmic scale),
not raw Hz, since that's how pitch distance is actually perceived -- the
same Hz gap means very different things in a low register vs a high one.

Reads:
  - vocal_db.csv      (built by extract_vocal_features.py)
  - user_profile.json (saved automatically by vocal_range_test_app.py)

Writes:
  - recommendations.csv (full ranked list)

Usage:
    python match_songs.py
"""

import csv
import json
import math
from pathlib import Path

# --- scoring weights (must sum to 1.0) ---
W_RANGE = 0.45
W_COMFORT = 0.30
W_TIMBRE = 0.15
W_VIBRATO = 0.10

TOP_N = 10


def hz_to_semitones(hz: float) -> float:
    """Convert Hz to a semitone scale (A4 = 440 Hz as the reference point),
    matching the standard MIDI note-number convention."""
    return 12 * math.log2(hz / 440.0) + 69


def score_song(user: dict, song: dict) -> dict:
    user_low_st = hz_to_semitones(user["lowest_hz"])
    user_high_st = hz_to_semitones(user["highest_hz"])
    song_low_st = hz_to_semitones(song["lowest_hz"])
    song_high_st = hz_to_semitones(song["highest_hz"])

    # How many semitones does the song demand beyond what the user can reach?
    low_excess = max(0.0, user_low_st - song_low_st)
    high_excess = max(0.0, song_high_st - user_high_st)
    # A full octave (12 semitones) of combined excess zeroes the score out;
    # smaller gaps fall off linearly (a song just out of reach by a tone or
    # two still scores reasonably, since octave-transposing it is trivial).
    range_score = max(0.0, 1 - (low_excess + high_excess) / 12.0)

    tessitura_center_st = (
        hz_to_semitones(song["tessitura_low_hz"]) + hz_to_semitones(song["tessitura_high_hz"])
    ) / 2
    comfort_distance = abs(hz_to_semitones(user["comfortable_hz"]) - tessitura_center_st)
    # A perfect-fifth (7 semitones) of distance zeroes this out -- comfort
    # fit is judged more strictly than raw reachability.
    comfort_score = max(0.0, 1 - comfort_distance / 7.0)

    if user.get("timbre_centroid_hz") and song.get("timbre_centroid_hz"):
        bigger = max(user["timbre_centroid_hz"], song["timbre_centroid_hz"])
        percent_diff = abs(user["timbre_centroid_hz"] - song["timbre_centroid_hz"]) / bigger
        timbre_score = max(0.0, 1 - percent_diff * 2)
    else:
        timbre_score = 0.5  # neutral when timbre data is missing on either side

    vibrato_score = 1.0 if user.get("vibrato_detected") == song.get("vibrato_detected") else 0.3

    total = (
        W_RANGE * range_score
        + W_COMFORT * comfort_score
        + W_TIMBRE * timbre_score
        + W_VIBRATO * vibrato_score
    )

    return {
        "total": round(total, 3),
        "range_score": round(range_score, 2),
        "comfort_score": round(comfort_score, 2),
        "timbre_score": round(timbre_score, 2),
        "vibrato_score": round(vibrato_score, 2),
    }


def load_song_db(path: Path) -> list:
    songs = []
    skipped = 0
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                songs.append({
                    "filename": row["filename"],
                    "lowest_hz": float(row["lowest_hz"]),
                    "highest_hz": float(row["highest_hz"]),
                    "tessitura_low_hz": float(row["tessitura_low_hz"]),
                    "tessitura_high_hz": float(row["tessitura_high_hz"]),
                    "lowest_note": row.get("lowest_note", ""),
                    "highest_note": row.get("highest_note", ""),
                    "timbre_centroid_hz": float(row["timbre_centroid_hz"]) if row.get("timbre_centroid_hz") else None,
                    "vibrato_detected": row.get("vibrato_detected", "False").strip().lower() == "true",
                })
            except (KeyError, ValueError):
                skipped += 1
    if skipped:
        print(f"(skipped {skipped} row(s) with missing/invalid data)")
    return songs


def load_user_profile(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def main():
    db_path = Path("vocal_db.csv")
    profile_path = Path("user_profile.json")

    if not db_path.exists():
        print(f"Couldn't find {db_path} in the current folder. Run extract_vocal_features.py first.")
        return
    if not profile_path.exists():
        print(f"Couldn't find {profile_path} in the current folder. Run vocal_range_test_app.py and complete a test first.")
        return

    user = load_user_profile(profile_path)
    songs = load_song_db(db_path)

    if not songs:
        print("Song database is empty -- nothing to match against.")
        return

    print(f"Matching against {len(songs)} song(s) in the database...\n")

    results = []
    for song in songs:
        scores = score_song(user, song)
        results.append({**song, **scores})

    results.sort(key=lambda r: r["total"], reverse=True)

    print(f"Top {min(TOP_N, len(results))} song matches for your voice:\n")
    for i, r in enumerate(results[:TOP_N], 1):
        print(f"{i}. {r['filename']}  —  {r['total'] * 100:.0f}% match")
        print(
            f"   range {r['range_score'] * 100:.0f}%  ·  comfort {r['comfort_score'] * 100:.0f}%  "
            f"·  timbre {r['timbre_score'] * 100:.0f}%  ·  style {r['vibrato_score'] * 100:.0f}%"
        )

    out_path = Path("recommendations.csv")
    fieldnames = [
        "filename", "total", "range_score", "comfort_score", "timbre_score", "vibrato_score",
        "lowest_note", "highest_note",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"\nFull ranked list ({len(results)} songs) saved to {out_path}")


if __name__ == "__main__":
    main()