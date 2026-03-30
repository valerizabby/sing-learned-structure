"""
Text2Prefix: текстовый промт → piano roll [T, 128] для SingLS.

Pipeline:
    text → MusicGen → wav → Basic Pitch → MIDI → piano roll

Использование:
    gen = Text2Prefix(model_size="small")
    piano_roll, tempo, num_beats = gen.generate(
        prompt  = "gentle piano melody in C major",
        n_bars  = 8,
        tempo   = 120.0,
    )
"""

import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import librosa


# ---------------------------------------------------------------------------
# Константы, согласованные с SingLS
# ---------------------------------------------------------------------------

BEATS_PER_BAR   = 4       # 4/4 размер
NOTE_MIN        = 20      # нижняя граница piano roll в SingLS
NOTE_MAX        = 108     # верхняя граница
MIDI_NOTES      = 128
MUSICGEN_SR     = 32000   # частота дискретизации MusicGen
BINARIZE_THRESH = 0.5     # порог бинаризации piano roll


class Text2Prefix:
    """
    Генерирует piano roll из текстового промта.

    Параметры:
        model_size : "small" (~300 MB) или "medium" (~1.5 GB)
        device     : "cpu" / "cuda" / "mps" / None (авто)
    """

    def __init__(self, model_size: str = "small", device: Optional[str] = None):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        self.model_size = model_size
        self._musicgen = None   # ленивая загрузка

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt:  str,
        n_bars:  int   = 8,
        tempo:   Optional[float] = 120.0,
    ) -> tuple[torch.Tensor, float, int]:
        """
        Генерирует piano roll из текстового промта.

        Args:
            prompt : текстовое описание музыки
            n_bars : желаемая длина в барах (по умолчанию 8)
            tempo  : темп в BPM; None = авто-определение из аудио

        Returns:
            piano_roll : FloatTensor [T, 128], бинаризованный
            tempo      : float, BPM
            num_beats  : int, количество долей (= T при fs=tempo/60)
        """
        # 1. Генерируем аудио
        duration_sec = self._bars_to_seconds(n_bars, tempo or 120.0)
        audio_mono = self._generate_audio(prompt, duration_sec)  # np [N]

        # 2. Определяем темп
        if tempo is None:
            tempo = self._detect_tempo(audio_mono)
            print(f"  Detected tempo: {tempo:.1f} BPM")

        # 3. Транскрибируем в MIDI
        midi = self._transcribe(audio_mono)

        # 4. Строим piano roll в формате SingLS
        piano_roll, num_beats = self._midi_to_piano_roll(midi, tempo, n_bars)

        return piano_roll, float(tempo), num_beats

    # ------------------------------------------------------------------
    # Шаг 1: MusicGen → audio
    # ------------------------------------------------------------------

    def _load_musicgen(self):
        """Ленивая загрузка MusicGen (тяжёлая модель)."""
        if self._musicgen is not None:
            return
        # audiocraft использует torch.autocast, который не поддерживает mps.
        # Загружаем на cpu; для cuda autocast работает штатно.
        load_device = "cpu" if self.device == "mps" else self.device
        print(f"Loading MusicGen-{self.model_size} on {load_device}...")
        from audiocraft.models import MusicGen
        self._musicgen = MusicGen.get_pretrained(
            f"facebook/musicgen-{self.model_size}",
            device=load_device,
        )
        print("  MusicGen loaded.")

    def _generate_audio(self, prompt: str, duration_sec: float) -> np.ndarray:
        """
        Генерирует аудио по тексту.
        Возвращает mono numpy array (float32, 32kHz).
        """
        self._load_musicgen()
        self._musicgen.set_generation_params(duration=duration_sec)
        # MusicGen возвращает Tensor [batch, channels, samples]
        with torch.no_grad():
            wav = self._musicgen.generate([prompt])  # [1, C, N]

        # Stereo → mono
        audio = wav[0].mean(dim=0).cpu().numpy().astype(np.float32)
        return audio

    # ------------------------------------------------------------------
    # Шаг 2: Basic Pitch → MIDI
    # ------------------------------------------------------------------

    def _transcribe(self, audio_mono: np.ndarray):
        """
        Транскрибирует аудио в MIDI с помощью Basic Pitch.
        Возвращает pretty_midi.PrettyMIDI.
        """
        from basic_pitch.inference import predict
        from basic_pitch import ICASSP_2022_MODEL_PATH

        # Basic Pitch принимает путь к файлу или numpy array + sr
        # Используем временный wav-файл для надёжности
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        sf.write(tmp_path, audio_mono, MUSICGEN_SR)

        model_output, midi_data, note_events = predict(tmp_path)

        Path(tmp_path).unlink(missing_ok=True)
        return midi_data  # pretty_midi.PrettyMIDI

    # ------------------------------------------------------------------
    # Шаг 3: MIDI → piano roll [T, 128]
    # ------------------------------------------------------------------

    def _midi_to_piano_roll(
        self,
        midi,
        tempo: float,
        n_bars: int,
    ) -> tuple[torch.Tensor, int]:
        """
        Конвертирует MIDI в piano roll формата SingLS.

        SingLS:
          - fs = tempo / 60  фреймов в секунду (1 фрейм = 1 доля)
          - T  = num_beats   (= n_bars × BEATS_PER_BAR)
          - shape [T, 128], float32, бинаризован

        Returns:
            piano_roll : Tensor [T, 128]
            num_beats  : int
        """
        fs           = tempo / 60.0          # фреймов в секунду
        num_beats    = n_bars * BEATS_PER_BAR
        duration_sec = num_beats / fs

        # pretty_midi: get_piano_roll(fs) → [128, frames]
        pr = midi.get_piano_roll(fs=fs)      # [128, frames_raw]

        # Обрезаем / дополняем до нужной длины
        target_frames = num_beats
        if pr.shape[1] >= target_frames:
            pr = pr[:, :target_frames]
        else:
            pad = np.zeros((128, target_frames - pr.shape[1]), dtype=pr.dtype)
            pr = np.concatenate([pr, pad], axis=1)

        # Транспонируем → [T, 128]
        pr = pr.T.astype(np.float32)         # [T, 128]

        # Бинаризуем
        pr = (pr > BINARIZE_THRESH).astype(np.float32)

        # Зануляем ноты вне диапазона SingLS [NOTE_MIN, NOTE_MAX]
        pr[:, :NOTE_MIN]    = 0.0
        pr[:, NOTE_MAX + 1:] = 0.0

        return torch.from_numpy(pr), num_beats

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def _bars_to_seconds(n_bars: int, tempo: float) -> float:
        """Длительность n_bars баров в секундах при заданном tempo."""
        beats = n_bars * BEATS_PER_BAR
        return beats / (tempo / 60.0)

    @staticmethod
    def _detect_tempo(audio_mono: np.ndarray, sr: int = MUSICGEN_SR) -> float:
        """Определяет темп из аудио через librosa."""
        tempo, _ = librosa.beat.beat_track(y=audio_mono, sr=sr)
        return float(np.atleast_1d(tempo)[0])
