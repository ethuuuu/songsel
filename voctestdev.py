import sys
import time
import json
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import parselmouth
import librosa
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QPushButton, QWidget, QLabel
from PyQt5.QtGui import QPainter, QPen, QColor, QPainterPath, QFont, QLinearGradient
from PyQt5.QtCore import QTimer, Qt

# --- Configuration ---
SAMPLE_RATE = 44100
CHANNELS = 1
CHUNK_SIZE = 1024

PITCH_FLOOR = 65     # ~C2, matches the range used on the song-database extraction side
PITCH_CEILING = 1000  # ~B5, covers high voices that a default 600 Hz ceiling would clip

MIN_PHASE_SECONDS = 0.5  # guards against an accidental double-click creating an empty segment

# Guided test script: self-paced, no fixed duration. The order mirrors how a
# vocal coach actually tests range: settle into a comfortable note first,
# then explore downward, then explore upward. The user advances each phase
# manually (button or spacebar) whenever they're ready.
PHASES = [
    {"name": "hold", "label": "Hold a comfortable, steady note", "color": (142, 68, 173)},
    {"name": "low",  "label": "Slide DOWN slowly, exploring as low as you can go, then hold your lowest note", "color": (69, 123, 157)},
    {"name": "high", "label": "Slide UP slowly, exploring as high as you can go, then hold your highest note", "color": (230, 126, 60)},
]


def _extract_pitch_values(audio_segment, sr):
    """Run Praat pitch tracking on a raw numpy audio segment, return voiced Hz values."""
    if len(audio_segment) < sr * 0.1:  # too short to analyze meaningfully
        return np.array([])
    sound = parselmouth.Sound(audio_segment.astype(np.float64), sampling_frequency=sr)
    pitch = sound.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
    values = pitch.selected_array["frequency"]
    return values[values > 0]


def analyze_vocal_profile(full_audio, sr, phase_bounds):
    """The DSP Engine: analyzes each phase of the guided test separately.
    - hold phase -> pitch stability (jitter/shimmer) and timbre, since those
      measurements need a steady sustained tone to be meaningful
    - low/high phases -> the actual extremes of the person's range
    """
    hold_audio = full_audio[phase_bounds["hold"][0]:phase_bounds["hold"][1]]
    low_audio = full_audio[phase_bounds["low"][0]:phase_bounds["low"][1]]
    high_audio = full_audio[phase_bounds["high"][0]:phase_bounds["high"][1]]

    hold_pitch_vals = _extract_pitch_values(hold_audio, sr)
    if len(hold_pitch_vals) == 0:
        return {"Error": "No distinct voice detected during the hold phase. Try again, singing a bit louder."}

    mean_pitch = np.mean(hold_pitch_vals)

    hold_sound = parselmouth.Sound(hold_audio.astype(np.float64), sampling_frequency=sr)
    point_process = parselmouth.praat.call(hold_sound, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([hold_sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    jitter = 0 if np.isnan(jitter) else jitter
    shimmer = 0 if np.isnan(shimmer) else shimmer

    spectral_centroids = librosa.feature.spectral_centroid(y=hold_audio.astype(np.float32), sr=sr)[0]
    mean_centroid = np.mean(spectral_centroids)

    low_vals = _extract_pitch_values(low_audio, sr)
    high_vals = _extract_pitch_values(high_audio, sr)

    # 5th/95th percentile rather than raw min/max, so a single stray
    # pitch-tracking glitch doesn't distort the reported range.
    lowest_hz = np.percentile(low_vals, 5) if len(low_vals) > 5 else (np.min(low_vals) if len(low_vals) else mean_pitch)
    highest_hz = np.percentile(high_vals, 95) if len(high_vals) > 5 else (np.max(high_vals) if len(high_vals) else mean_pitch)

    vibrato_detected = bool(_detect_vibrato_for_user(hold_pitch_vals, sr))

    display = {
        "Lowest Note": f"{librosa.hz_to_note(lowest_hz)}  ({lowest_hz:.1f} Hz)",
        "Highest Note": f"{librosa.hz_to_note(highest_hz)}  ({highest_hz:.1f} Hz)",
        "Comfortable Pitch (Hold)": f"{librosa.hz_to_note(mean_pitch)}  ({mean_pitch:.1f} Hz)",
        "Pitch Jitter (Instability)": f"{jitter * 100:.2f}%",
        "Volume Shimmer (Breathiness)": f"{shimmer * 100:.2f}%",
        "Timbre Brightness": f"{mean_centroid:.1f} Hz",
    }

    raw = {
        "lowest_hz": round(float(lowest_hz), 2),
        "highest_hz": round(float(highest_hz), 2),
        "comfortable_hz": round(float(mean_pitch), 2),
        "jitter_percent": round(float(jitter) * 100, 3),
        "shimmer_percent": round(float(shimmer) * 100, 3),
        "timbre_centroid_hz": round(float(mean_centroid), 1),
        "vibrato_detected": vibrato_detected,
    }

    return {"display": display, "raw": raw}


def _detect_vibrato_for_user(f0_voiced, sr):
    """Same 4-7 Hz periodicity heuristic used on the song-extraction side,
    applied to the hold-phase pitch contour."""
    hop_rate = sr / 512  # approximate frame rate of the underlying pitch values
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


class WaveformWidget(QWidget):
    """An organic, fluid audio ribbon that recolors itself per test phase,
    plus a live instruction + elapsed-time overlay."""

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(280)
        self.points_count = 60
        self.audio_data = np.zeros(self.points_count)
        self.time_counter = 0.0

        self.show_target = False
        self.current_phase_color = QColor(142, 68, 173)
        self.phase_text = ""
        self.phase_elapsed = 0.0

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_wave)
        self.timer.start(16)

    def update_audio(self, data):
        if len(data) == 0:
            return
        chunks = np.array_split(data, self.points_count)
        downsampled = np.array([np.mean(c) for c in chunks])
        kernel = np.ones(5) / 5
        smoothed = np.convolve(downsampled, kernel, mode="same")
        self.audio_data = (self.audio_data * 0.6) + (smoothed * 0.4)

    def set_phase(self, color_rgb, text, elapsed):
        self.current_phase_color = QColor(*color_rgb)
        self.phase_text = text
        self.phase_elapsed = elapsed

    def update_wave(self):
        self.time_counter += 0.04
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        mid_y = height / 2

        painter.fillRect(self.rect(), QColor(14, 14, 18))

        if self.show_target:
            self.draw_phase_overlay(painter, width, height)

        base = self.current_phase_color
        gradient_shift = np.sin(self.time_counter * 0.5) * (width * 0.3)
        gradient = QLinearGradient(0 + gradient_shift, 0, width + gradient_shift, height)
        gradient.setColorAt(0.0, QColor(25, 42, 86))
        gradient.setColorAt(0.5, base)
        gradient.setColorAt(1.0, QColor(232, 67, 147))

        self.draw_ribbon(painter, width, mid_y, gradient, opacity=0.3, phase_offset=1.5, scale=0.7)
        self.draw_ribbon(painter, width, mid_y, gradient, opacity=1.0, phase_offset=0.0, scale=1.0)

    def draw_phase_overlay(self, painter, width, height):
        painter.setPen(QColor(255, 255, 255, 170))
        font = QFont("Arial", 22, QFont.Bold)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(20, -50, -20, -50), Qt.AlignCenter, self.phase_text)

        painter.setPen(QColor(255, 255, 255, 110))
        font_small = QFont("Arial", 14)
        painter.setFont(font_small)
        painter.drawText(self.rect().adjusted(0, 30, 0, 30), Qt.AlignCenter, f"{self.phase_elapsed:.1f}s — press Next or Space when ready")

    def draw_ribbon(self, painter, width, mid_y, gradient, opacity, phase_offset, scale):
        pen = QPen(gradient, 4 if opacity == 1.0 else 2)
        painter.setPen(pen)
        painter.setOpacity(opacity)

        path = QPainterPath()
        step = width / (self.points_count - 1)

        for i in range(self.points_count):
            x = i * step
            mic_signal = self.audio_data[i] * (self.height() * 0.8) * scale
            ambient_wave = 12 * np.sin(i * 0.15 + self.time_counter * 3 + phase_offset)
            ambient_sub = 6 * np.cos(i * 0.3 - self.time_counter * 1.5)

            ambient_multiplier = 1.0 + (np.abs(mic_signal) * 2.0)
            total_ambient = (ambient_wave + ambient_sub) * ambient_multiplier

            y = mid_y - (mic_signal + total_ambient)

            if i == 0:
                path.moveTo(x, y)
            else:
                prev_x = (i - 1) * step
                cx = (prev_x + x) / 2
                path.quadTo(prev_x, path.currentPosition().y(), cx, (path.currentPosition().y() + y) / 2)
                path.lineTo(x, y)

        painter.drawPath(path)
        painter.setOpacity(1.0)


class VoiceTestApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Vocal AI Profiler")
        self.resize(850, 750)
        self.setStyleSheet("background-color: #0E0E12; color: white; font-family: Arial, Helvetica, sans-serif;")

        self.is_recording = False
        self.recorded_frames = []
        self.stream = None
        self.recording_start_time = None
        self.current_phase_index = -1
        self.current_phase_start_elapsed = 0.0
        self.phase_elapsed_bounds = []  # list of (start_sec, end_sec) per completed phase

        main_widget = QWidget()
        layout = QVBoxLayout()

        self.instruction_label = QLabel(
            "Click start, then take your time on each phase. Press Next (or Space) whenever you're ready to move on."
        )
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setStyleSheet("font-size: 18px; color: #E0E0E0; margin: 15px;")
        layout.addWidget(self.instruction_label)

        self.progress_label = QLabel("")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setStyleSheet("font-size: 13px; color: #8A8A93; margin-bottom: 10px;")
        layout.addWidget(self.progress_label)

        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        self.record_btn = QPushButton("🔴 Start Vocal Test")
        self.record_btn.setStyleSheet("""
            QPushButton { background-color: #E63946; color: white; font-size: 16px; font-weight: bold; padding: 18px; border-radius: 8px; }
            QPushButton:hover { background-color: #FF4D5A; }
            QPushButton:disabled { background-color: #4A4A52; color: #8A8A93; }
        """)
        self.record_btn.clicked.connect(self.handle_primary_button)
        layout.addWidget(self.record_btn)

        self.results_panel = QLabel("")
        self.results_panel.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.results_panel.setStyleSheet("""
            background-color: #1A1A20;
            padding: 25px;
            border-radius: 10px;
            border: 1px solid #2A2A35;
            font-size: 16px;
            color: #A8DADC;
            line-height: 1.8;
        """)
        self.results_panel.hide()
        layout.addWidget(self.results_panel)

        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        self.ui_timer = QTimer()
        self.ui_timer.timeout.connect(self.refresh_phase_display)

        self.start_mic_listening()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Space, Qt.Key_Return, Qt.Key_Enter) and self.record_btn.isEnabled():
            self.record_btn.click()
        else:
            super().keyPressEvent(event)

    def audio_callback(self, indata, frames, time_info, status):
        audio_chunk = indata[:, 0]
        self.waveform.update_audio(audio_chunk.copy())
        if self.is_recording:
            self.recorded_frames.append(indata.copy())

    def start_mic_listening(self):
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback, blocksize=CHUNK_SIZE)
        self.stream.start()

    def handle_primary_button(self):
        if not self.is_recording:
            self.start_test()
        else:
            self.advance_phase()

    def start_test(self):
        self.is_recording = True
        self.recorded_frames = []
        self.recording_start_time = time.time()
        self.current_phase_index = 0
        self.current_phase_start_elapsed = 0.0
        self.phase_elapsed_bounds = []

        self.waveform.show_target = True
        self.results_panel.hide()
        self.apply_phase_ui(PHASES[0])
        self.ui_timer.start(100)

    def advance_phase(self):
        now_elapsed = time.time() - self.recording_start_time
        if now_elapsed - self.current_phase_start_elapsed < MIN_PHASE_SECONDS:
            return  # too soon, likely an accidental double-press; ignore

        self.phase_elapsed_bounds.append((self.current_phase_start_elapsed, now_elapsed))
        self.current_phase_start_elapsed = now_elapsed
        self.current_phase_index += 1

        if self.current_phase_index >= len(PHASES):
            self.finish_recording()
        else:
            self.apply_phase_ui(PHASES[self.current_phase_index])

    def apply_phase_ui(self, phase):
        is_last = self.current_phase_index == len(PHASES) - 1
        self.instruction_label.setText(phase["label"])
        self.progress_label.setText(f"Step {self.current_phase_index + 1} of {len(PHASES)}")
        self.waveform.set_phase(phase["color"], phase["label"], 0.0)
        self.record_btn.setText("✅ Finish Test" if is_last else "➡️ Next Phase")
        self.record_btn.setStyleSheet("background-color: #457B9D; padding: 18px; border-radius: 8px; font-weight: bold; font-size: 16px;")

    def refresh_phase_display(self):
        if not self.is_recording:
            return
        now_elapsed = time.time() - self.recording_start_time
        elapsed_in_phase = now_elapsed - self.current_phase_start_elapsed
        phase = PHASES[self.current_phase_index]
        self.waveform.set_phase(phase["color"], phase["label"], elapsed_in_phase)

    def finish_recording(self):
        self.is_recording = False
        self.ui_timer.stop()
        self.waveform.show_target = False

        self.record_btn.setText("⚙️ Processing ML Features...")
        self.record_btn.setStyleSheet("background-color: #F4A261; padding: 18px; border-radius: 8px; font-weight: bold; font-size: 16px; color: black;")
        self.record_btn.setEnabled(False)
        self.instruction_label.setText("Analyzing vocal stability, range, and timbre...")

        QTimer.singleShot(100, self.save_and_process_audio)

    def save_and_process_audio(self):
        if self.recorded_frames:
            audio_data = np.concatenate(self.recorded_frames, axis=0)[:, 0]
            file_name = "user_vocal_test.wav"
            write(file_name, SAMPLE_RATE, audio_data)

            phase_bounds = {}
            for i, (start_sec, end_sec) in enumerate(self.phase_elapsed_bounds):
                start = min(int(start_sec * SAMPLE_RATE), len(audio_data))
                end = min(int(end_sec * SAMPLE_RATE), len(audio_data))
                phase_bounds[PHASES[i]["name"]] = (start, end)

            try:
                profile = analyze_vocal_profile(audio_data, SAMPLE_RATE, phase_bounds)

                if "Error" in profile:
                    self.instruction_label.setText(f"⚠️ {profile['Error']}")
                else:
                    display = profile["display"]
                    result_text = "<span style='color: #FFFFFF; font-size: 18px; font-weight: bold;'>Vocal Blueprint Extracted:</span><br><br>"
                    for key, value in display.items():
                        result_text += f"• <b>{key}:</b> <span style='color: #F1FAEE;'>{value}</span><br>"

                    self.results_panel.setText(result_text)
                    self.results_panel.show()

                    with open("user_profile.json", "w") as pf:
                        json.dump(profile["raw"], pf, indent=2)

                    self.instruction_label.setText("Profile mapped successfully. Saved to user_profile.json.")
                    self.progress_label.setText("")
            except Exception as e:
                self.instruction_label.setText(f"⚠️ Error processing audio: {e}")

            self.record_btn.setText("🔴 Restart Test")
            self.record_btn.setStyleSheet("background-color: #E63946; padding: 18px; border-radius: 8px; font-weight: bold; font-size: 16px;")
            self.record_btn.setEnabled(True)

    def closeEvent(self, event):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = VoiceTestApp()
    window.show()
    sys.exit(app.exec_())