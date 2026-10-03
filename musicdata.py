import sys
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

def analyze_vocal_profile(audio_file_path):
    """The DSP Engine: Extracts pitch, stability, and timbre."""
    sound = parselmouth.Sound(audio_file_path)
    
    # --- PITCH ---
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0] 
    
    if len(pitch_values) == 0:
        return {"Error": "No distinct voice detected. Please try singing louder."}
        
    mean_pitch = np.mean(pitch_values)
    min_pitch = np.min(pitch_values)
    max_pitch = np.max(pitch_values)
    
    # --- STABILITY ---
    point_process = parselmouth.praat.call(sound, "To PointProcess (periodic, cc)", 75, 500)
    # Note: 0, 0 added to fix the Praat minimum/maximum period parameter constraints
    jitter = parselmouth.praat.call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    shimmer = parselmouth.praat.call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    jitter = 0 if np.isnan(jitter) else jitter
    shimmer = 0 if np.isnan(shimmer) else shimmer
    
    # --- TIMBRE ---
    y, sr = librosa.load(audio_file_path)
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    mean_centroid = np.mean(spectral_centroids)
    
    return {
        "Vocal Range": f"{min_pitch:.1f} Hz  →  {max_pitch:.1f} Hz",
        "Target Accuracy (Mean)": f"{mean_pitch:.1f} Hz",
        "Pitch Jitter (Instability)": f"{jitter * 100:.2f}%",
        "Volume Shimmer (Breathiness)": f"{shimmer * 100:.2f}%",
        "Timbre Brightness": f"{mean_centroid:.1f} Hz"
    }

class WaveformWidget(QWidget):
    """An organic, fluid audio ribbon with a target 'Ghost Wave' for tone holding."""
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(280)
        self.points_count = 60
        self.audio_data = np.zeros(self.points_count)
        self.time_counter = 0.0
        
        self.show_target = False
        self.countdown_time = 3.0  
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_wave)
        self.timer.start(16) 

    def update_audio(self, data):
        if len(data) == 0:
            return
        chunks = np.array_split(data, self.points_count)
        downsampled = np.array([np.mean(c) for c in chunks])
        kernel = np.ones(5) / 5
        smoothed = np.convolve(downsampled, kernel, mode='same')
        self.audio_data = (self.audio_data * 0.6) + (smoothed * 0.4)

    def update_wave(self):
        self.time_counter += 0.04
        if self.show_target and self.countdown_time > 0:
            self.countdown_time -= 0.016 
            if self.countdown_time <= 0:
                self.countdown_time = 0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        mid_y = height / 2
        
        painter.fillRect(self.rect(), QColor(14, 14, 18))

        if self.show_target:
            self.draw_ghost_wave(painter, width, mid_y)
            self.draw_countdown(painter, width, height)

        gradient_shift = np.sin(self.time_counter * 0.5) * (width * 0.3)
        gradient = QLinearGradient(0 + gradient_shift, 0, width + gradient_shift, height)
        gradient.setColorAt(0.0, QColor(25, 42, 86))
        gradient.setColorAt(0.5, QColor(142, 68, 173))
        gradient.setColorAt(1.0, QColor(232, 67, 147))

        self.draw_ribbon(painter, width, mid_y, gradient, opacity=0.3, phase_offset=1.5, scale=0.7)
        self.draw_ribbon(painter, width, mid_y, gradient, opacity=1.0, phase_offset=0.0, scale=1.0)

    def draw_ghost_wave(self, painter, width, mid_y):
        pen = QPen(QColor(255, 255, 255, 70)) 
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine) 
        painter.setPen(pen)

        path = QPainterPath()
        step = width / (self.points_count - 1)
        
        for i in range(self.points_count):
            x = i * step
            y = mid_y - (30 * np.sin(i * 0.2 + self.time_counter * 2))
            
            if i == 0:
                path.moveTo(x, y)
            else:
                path.lineTo(x, y)
        painter.drawPath(path)

    def draw_countdown(self, painter, width, height):
        painter.setPen(QColor(255, 255, 255, 120)) 
        font = QFont("Arial", 54, QFont.Bold)
        painter.setFont(font)
        time_text = f"Hold Tone: {max(0, self.countdown_time):.1f}s"
        painter.drawText(self.rect(), Qt.AlignCenter, time_text)

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
        self.resize(850, 700)
        self.setStyleSheet("background-color: #0E0E12; color: white; font-family: Arial, Helvetica, sans-serif;")

        self.is_recording = False
        self.recorded_frames = []
        self.stream = None

        main_widget = QWidget()
        layout = QVBoxLayout()

        self.instruction_label = QLabel("Click start. Match the height of the ghost wave and hold a steady tone.")
        self.instruction_label.setAlignment(Qt.AlignCenter)
        self.instruction_label.setStyleSheet("font-size: 18px; color: #E0E0E0; margin: 15px;")
        layout.addWidget(self.instruction_label)

        self.waveform = WaveformWidget()
        layout.addWidget(self.waveform)

        self.record_btn = QPushButton("🔴 Start 3-Second Vocal Test")
        self.record_btn.setStyleSheet("""
            QPushButton { background-color: #E63946; color: white; font-size: 16px; font-weight: bold; padding: 18px; border-radius: 8px; }
            QPushButton:hover { background-color: #FF4D5A; }
            QPushButton:disabled { background-color: #4A4A52; color: #8A8A93; }
        """)
        self.record_btn.clicked.connect(self.toggle_recording)
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
        self.start_mic_listening()

    def audio_callback(self, indata, frames, time, status):
        audio_chunk = indata[:, 0]
        self.waveform.update_audio(audio_chunk.copy())
        if self.is_recording:
            self.recorded_frames.append(indata.copy())

    def start_mic_listening(self):
        self.stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=CHANNELS, callback=self.audio_callback, blocksize=CHUNK_SIZE)
        self.stream.start()

    def toggle_recording(self):
        if not self.is_recording:
            self.is_recording = True
            self.recorded_frames = []
            
            self.waveform.show_target = True
            self.waveform.countdown_time = 3.0
            
            self.record_btn.setText("⬛ Stop Early")
            self.record_btn.setStyleSheet("background-color: #457B9D; padding: 18px; border-radius: 8px; font-weight: bold; font-size: 16px;")
            self.instruction_label.setText("Recording... Match the ghost wave!")
            self.results_panel.hide()
            
            self.auto_stop_timer = QTimer()
            self.auto_stop_timer.setSingleShot(True)
            self.auto_stop_timer.timeout.connect(self.stop_recording)
            self.auto_stop_timer.start(3000) 
        else:
            if hasattr(self, 'auto_stop_timer') and self.auto_stop_timer.isActive():
                self.auto_stop_timer.stop()
            self.stop_recording()

    def stop_recording(self):
        self.is_recording = False
        self.waveform.show_target = False 
        
        self.record_btn.setText("⚙️ Processing ML Features...")
        self.record_btn.setStyleSheet("background-color: #F4A261; padding: 18px; border-radius: 8px; font-weight: bold; font-size: 16px; color: black;")
        self.record_btn.setEnabled(False)
        self.instruction_label.setText("Analyzing vocal stability and timbre...")
        
        # UI needs a tiny delay to update the button text before blocking the thread with DSP
        QTimer.singleShot(100, self.save_and_process_audio)

    def save_and_process_audio(self):
        if self.recorded_frames:
            audio_data = np.concatenate(self.recorded_frames, axis=0)
            file_name = "user_vocal_test.wav"
            write(file_name, SAMPLE_RATE, audio_data)
            
            try:
                profile = analyze_vocal_profile(file_name)
                
                if "Error" in profile:
                    self.instruction_label.setText(f"⚠️ {profile['Error']}")
                else:
                    result_text = "<span style='color: #FFFFFF; font-size: 18px; font-weight: bold;'>Vocal Blueprint Extracted:</span><br><br>"
                    for key, value in profile.items():
                        result_text += f"• <b>{key}:</b> <span style='color: #F1FAEE;'>{value}</span><br>"
                    
                    self.results_panel.setText(result_text)
                    self.results_panel.show()
                    self.instruction_label.setText("Profile mapped successfully.")
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