"""
Text2PrefixMIDI: text → piano roll [T, 128] через прямую генерацию MIDI.

Использует text2midi (amaai-lab/text2midi) — seq2seq трансформер:
  • Text encoder:  google/flan-t5-base  (заморожен)
  • MIDI decoder:  custom TransformerDecoder  (REMI+ токенизация)
  • Веса:          amaai-lab/text2midi / pytorch_model.bin
  • Вокабуляр:     amaai-lab/text2midi / vocab_remi.pkl

Pipeline:
    text  →  T5Tokenizer  →  Transformer.generate()
          →  REMI-токены  →  symusic.Score
          →  pretty_midi  →  piano roll [T, 128]

Преимущества перед MusicGen+BasicPitch:
    - Нет шума транскрипции (audio→MIDI)
    - Точный контроль темпа через REMI tempo-токены
    - Более лёгкие зависимости (нет audiocraft/soundfile)

Интерфейс идентичен Text2Prefix:
    gen = Text2PrefixMIDI()
    piano_roll, tempo, num_beats = gen.generate(
        prompt  = "gentle piano melody in C major",
        n_bars  = 8,
        tempo   = 120.0,
    )
"""

import pickle
import sys
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import pretty_midi

# ── Константы, согласованные с SingLS ──────────────────────────────────────

BEATS_PER_BAR   = 4
NOTE_MIN        = 20
NOTE_MAX        = 108
MIDI_NOTES      = 128
BINARIZE_THRESH = 0.5

# Параметры модели из README amaai-lab/text2midi
_MODEL_KWARGS = dict(
    d_model          = 768,
    nhead            = 8,
    max_len          = 2048,
    num_decoder_layers = 18,
    dim_feedforward  = 1024,
    use_moe          = False,
    num_experts      = 8,
)
_REPO_ID    = "amaai-lab/text2midi"
_T5_ID      = "google/flan-t5-base"

# REMI-токенов на такт.  text2midi в среднем использует ~100–150 токенов/такт.
# Слишком маленькое значение → ранний обрыв генерации → пустой prefix.
_TOKENS_PER_BAR = 120

# Внутреннее разрешение piano roll при извлечении из MIDI.
# SingLS использует fs = tempo/60 (≈2 fps при 120 BPM — 1 фрейм/бит).
# При таком разрешении восьмые/шестнадцатые ноты не попадают в бины.
# Решение: сэмплируем с высоким fs, потом max-pool до целевого разрешения.
_PIANO_ROLL_OVERSAMPLE = 8   # во сколько раз превышать целевое разрешение


class Text2PrefixMIDI:
    """
    Генерирует piano roll из текстового промта через прямую MIDI-генерацию.

    Параметры:
        device           : "cpu" / "cuda" / "mps" / None (авто)
        temperature      : температура сэмплирования (1.0 = без изменений)
        tokens_per_bar   : лимит MIDI-токенов на такт (влияет на max_len)
    """

    def __init__(
        self,
        device:         Optional[str]   = None,
        temperature:    float           = 1.0,
        tokens_per_bar: int             = _TOKENS_PER_BAR,
        use_half:       bool            = False,  # float16 (только CUDA)
        compile_model:  bool            = False,  # torch.compile (только CUDA, Python≥3.10)
    ):
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device         = device
        self.temperature    = temperature
        self.tokens_per_bar = tokens_per_bar
        # half precision имеет смысл только на CUDA; на MPS/CPU оставляем float32
        self.use_half       = use_half and (device == "cuda")
        self.compile_model  = compile_model and (device == "cuda")

        self._model         = None   # Transformer (lazy)
        self._t5_tokenizer  = None   # T5Tokenizer (lazy)
        self._midi_tokenizer = None  # vocab_remi pickle (lazy)

    # ── Публичный API ───────────────────────────────────────────────────────

    def generate(
        self,
        prompt:  str,
        n_bars:  int            = 8,
        tempo:   Optional[float] = 120.0,
    ) -> tuple:
        """
        Генерирует piano roll из текстового промта.

        Returns:
            piano_roll : FloatTensor [T, 128], бинаризованный
            tempo      : float, BPM
            num_beats  : int
        """
        self._load_model()

        if tempo is None:
            tempo = 120.0

        enriched = _build_prompt(prompt, int(tempo), n_bars)
        print(f"  Prompt: {enriched}")

        midi = self._generate_midi(enriched, n_bars)
        piano_roll, num_beats = _midi_to_piano_roll(midi, tempo, n_bars)
        return piano_roll, float(tempo), num_beats

    # ── Загрузка модели ─────────────────────────────────────────────────────

    def _load_model(self):
        """Ленивая загрузка text2midi (T5 encoder + REMI decoder)."""
        if self._model is not None:
            return

        from huggingface_hub import hf_hub_download
        from transformers import T5Tokenizer

        print(f"Loading text2midi on {self.device}...")

        # 1. Скачиваем vocab и веса модели
        model_path = hf_hub_download(repo_id=_REPO_ID, filename="pytorch_model.bin")
        vocab_path  = hf_hub_download(repo_id=_REPO_ID, filename="vocab_remi.pkl")
        arch_path   = hf_hub_download(repo_id=_REPO_ID, filename="transformer_model.py")

        # 2. Загружаем REMI-вокабуляр
        with open(vocab_path, "rb") as f:
            self._midi_tokenizer = pickle.load(f)
        # Патч: vocab_remi.pkl сохранён под старой версией miditok 3.x,
        # в которой TokenizerConfig имел дополнительные атрибуты.
        # miditok 3.0.6 их не создаёт — добавляем вручную.
        cfg = self._midi_tokenizer.config
        _COMPAT_PATCHES = {
            "use_velocities":             True,
            "use_note_duration_programs": [],   # должен быть итерируемым
            "default_note_duration":      0.5,  # доля бита (восьмая нота)
        }
        for attr, val in _COMPAT_PATCHES.items():
            if not hasattr(cfg, attr):
                setattr(cfg, attr, val)
        vocab_size = len(self._midi_tokenizer)
        print(f"  REMI vocab size: {vocab_size}")

        # 3. Импортируем Transformer из скачанного transformer_model.py
        arch_dir = str(Path(arch_path).parent)
        if arch_dir not in sys.path:
            sys.path.insert(0, arch_dir)
        from transformer_model import Transformer  # noqa: PLC0415

        # 4. Инициализируем модель с правильными параметрами (из README)
        self._model = Transformer(
            n_vocab = vocab_size,
            device  = self.device,
            **_MODEL_KWARGS,
        )
        state = torch.load(model_path, map_location=self.device, weights_only=False)
        self._model.load_state_dict(state)
        self._model.eval()

        if self.use_half:
            self._model = self._model.half()
            print("  Model converted to float16.")

        if self.compile_model:
            self._model = torch.compile(self._model)
            print("  Model compiled with torch.compile.")

        print("  Transformer weights loaded.")

        # 5. T5 tokenizer для текста
        self._t5_tokenizer = T5Tokenizer.from_pretrained(_T5_ID)
        print("  T5 tokenizer loaded.")

    # ── Генерация MIDI ──────────────────────────────────────────────────────

    def _generate_midi(self, prompt: str, n_bars: int) -> pretty_midi.PrettyMIDI:
        """Запускает авторегрессивную генерацию MIDI-токенов."""
        max_len = n_bars * self.tokens_per_bar

        # Токенизируем текст (T5)
        inputs = self._t5_tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        input_ids      = nn.utils.rnn.pad_sequence(
            inputs.input_ids,      batch_first=True, padding_value=0
        ).to(self.device)
        attention_mask = nn.utils.rnn.pad_sequence(
            inputs.attention_mask, batch_first=True, padding_value=0
        ).to(self.device)

        # Генерируем MIDI-токены
        with torch.no_grad():
            output = self._model.generate(
                input_ids,
                attention_mask,
                max_len=max_len,
                temperature=self.temperature,
            )

        output_list = output[0].tolist()
        return self._tokens_to_midi(output_list)

    def _tokens_to_midi(self, token_ids: list) -> pretty_midi.PrettyMIDI:
        """
        Конвертирует список REMI-токенов в pretty_midi.PrettyMIDI.

        r_tok.decode(flat_list) → symusic.ScoreTick → dump_midi(path) → pretty_midi.
        """
        try:
            score = self._midi_tokenizer.decode(token_ids)   # flat list, не [[...]]
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
                tmp = f.name
            score.dump_midi(tmp)
            midi = pretty_midi.PrettyMIDI(tmp)
            Path(tmp).unlink(missing_ok=True)
            return midi
        except Exception as e:
            print(f"  [text2midi] MIDI decode error: {e}")
            print("  Returning empty MIDI (fallback).")
            return pretty_midi.PrettyMIDI(initial_tempo=120.0)


# ── Вспомогательные функции (module-level) ─────────────────────────────────

def _build_prompt(prompt: str, tempo: int, n_bars: int) -> str:
    """
    Обогащает промт темпом и длиной.
    text2midi обучался на подписях MidiCaps, включающих BPM и структурные описания.
    """
    additions = []
    if "bpm" not in prompt.lower() and "tempo" not in prompt.lower():
        additions.append(f"{tempo} BPM")
    if "bar" not in prompt.lower():
        additions.append(f"{n_bars} bars")
    if additions:
        return f"{prompt}, {', '.join(additions)}"
    return prompt


def _midi_to_piano_roll(
    midi:   pretty_midi.PrettyMIDI,
    tempo:  float,
    n_bars: int,
) -> tuple:
    """
    Конвертирует MIDI в piano roll формата SingLS.

    SingLS:
      - fs_target = tempo / 60  фреймов/сек (1 фрейм = 1 доля)
      - T  = n_bars × BEATS_PER_BAR
      - shape [T, 128], float32, бинаризован
      - ноты вне [NOTE_MIN, NOTE_MAX] обнуляются

    Проблема с наивным подходом: при fs=2 (120 BPM) восьмые/шестнадцатые ноты
    длиннее 1 фрейма не попадают в бины → пустой piano roll.
    Решение: сэмплируем с fs_high = fs_target * OVERSAMPLE, затем max-pool.

    Returns:
        piano_roll : Tensor [T, 128]
        num_beats  : int
    """
    fs_target = tempo / 60.0
    fs_high   = fs_target * _PIANO_ROLL_OVERSAMPLE
    num_beats = n_bars * BEATS_PER_BAR
    num_frames_high = num_beats * _PIANO_ROLL_OVERSAMPLE   # целевое кол-во фреймов высокого разрешения

    pr_high = midi.get_piano_roll(fs=fs_high)   # [128, frames_raw]

    # Обрезаем или дополняем до нужной длины
    if pr_high.shape[1] >= num_frames_high:
        pr_high = pr_high[:, :num_frames_high]
    else:
        pad = np.zeros((128, num_frames_high - pr_high.shape[1]), dtype=pr_high.dtype)
        pr_high = np.concatenate([pr_high, pad], axis=1)

    # Max-pool: из каждых _OVERSAMPLE фреймов берём максимум → [128, T]
    pr_high = pr_high.reshape(128, num_beats, _PIANO_ROLL_OVERSAMPLE)
    pr = pr_high.max(axis=2)               # [128, T]

    pr = pr.T.astype(np.float32)           # [T, 128]
    pr = (pr > BINARIZE_THRESH).astype(np.float32)
    pr[:, :NOTE_MIN]     = 0.0
    pr[:, NOTE_MAX + 1:] = 0.0

    # Детектируем реальный конец контента: последний бит с хотя бы одной нотой.
    # text2midi может завершиться раньше max_len (EOS), тогда остаток — тишина.
    # Обрезаем до реального контента чтобы T_prefix точно соответствовал музыке.
    active_beats = np.where(pr.sum(axis=1) > 0)[0]
    if active_beats.size > 0:
        actual_beats = int(active_beats[-1]) + 1
        if actual_beats < num_beats:
            print(
                f"  [!] text2midi сгенерировал только {actual_beats}/{num_beats} битов "
                f"({actual_beats // BEATS_PER_BAR:.1f}/{n_bars} баров). "
                f"Триммируем prefix до реального контента."
            )
            pr        = pr[:actual_beats]
            num_beats = actual_beats
    else:
        print("  [!] Prefix полностью пустой — возможно, text2midi не сгенерировал нот.")

    # Диагностика
    density      = pr.mean()
    active_notes = int((pr.sum(axis=0) > 0).sum())
    print(f"  Prefix density  : {density:.4f}  (ожидается 0.05–0.15)")
    print(f"  Active notes    : {active_notes}/128  (ожидается 20–60)")
    print(f"  Prefix shape    : {pr.shape}  ({pr.shape[0] // BEATS_PER_BAR} bars)")

    return torch.from_numpy(pr), num_beats

