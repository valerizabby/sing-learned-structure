"""
AffinitySSM: построение SSM из сегментного плана через матрицу музыкального сходства.

Использование:
    # Из сохранённой матрицы (empirical):
    ssm_builder = AffinitySSM.from_checkpoint("checkpoints/affinity_matrix.pt")

    # Из теоретических значений (fixed):
    ssm_builder = AffinitySSM.fixed()

    # Построить SSM для трека:
    segment_plan = [(1, 16), (2, 16), (1, 16), (2, 16)]  # [(label_id, n_bars)]
    ssm = ssm_builder.build(segment_plan, ssm_size=64)    # Tensor [64, 64]
"""

from pathlib import Path
from typing import List, Tuple, Optional

import torch
import torch.nn.functional as F
import numpy as np

LABEL_NAMES = {
    0: "intro",
    1: "verse",
    2: "chorus",
    3: "bridge",
    4: "instr",
    5: "outro",
    6: "other",
    7: "unknown",
}
N_LABELS = len(LABEL_NAMES)

# Теоретическая матрица A, основанная на теории музыкальной формы.
# Принципы:
#   - диагональ = 1.0 (секция идентична себе)
#   - verse ↔ chorus = 0.7 (оба несут основной материал в одной тональности)
#   - intro ↔ outro = 0.6 (обрамляют трек, часто схожи тематически)
#   - bridge: низкое сходство (0.2–0.3) — контрастная секция
#   - instr: среднее (0.5) — инструментальная версия verse/chorus
#   - unknown: нейтральное (0.3)
A_FIXED_VALUES = [
    # intro verse chorus bridge instr outro other unknown
    [1.00, 0.60, 0.40,  0.20, 0.50, 0.60, 0.30, 0.30],  # intro
    [0.60, 1.00, 0.70,  0.25, 0.50, 0.35, 0.30, 0.30],  # verse
    [0.40, 0.70, 1.00,  0.30, 0.55, 0.45, 0.30, 0.30],  # chorus
    [0.20, 0.25, 0.30,  1.00, 0.25, 0.20, 0.25, 0.25],  # bridge
    [0.50, 0.50, 0.55,  0.25, 1.00, 0.40, 0.30, 0.30],  # instr
    [0.60, 0.35, 0.45,  0.20, 0.40, 1.00, 0.30, 0.30],  # outro
    [0.30, 0.30, 0.30,  0.25, 0.30, 0.30, 1.00, 0.30],  # other
    [0.30, 0.30, 0.30,  0.25, 0.30, 0.30, 0.30, 1.00],  # unknown
]


def _gaussian_blur_2d(ssm: torch.Tensor, sigma: float) -> torch.Tensor:
    """Применяет 2D Gaussian blur к SSM [T, T]."""
    if sigma <= 0:
        return ssm
    # Радиус ядра
    radius = max(1, int(3 * sigma))
    size = 2 * radius + 1
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d.unsqueeze(0) * kernel_1d.unsqueeze(1)  # [size, size]
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)               # [1,1,size,size]

    ssm_4d = ssm.unsqueeze(0).unsqueeze(0)  # [1,1,T,T]
    blurred = F.conv2d(ssm_4d, kernel_2d, padding=radius)
    return blurred.squeeze(0).squeeze(0).clamp(0.0, 1.0)


def _resize_ssm(ssm: torch.Tensor, size: int) -> torch.Tensor:
    """Resize [T,T] → [size,size] через bilinear интерполяцию."""
    return F.interpolate(
        ssm.unsqueeze(0).unsqueeze(0).float(),
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).squeeze(0)


class AffinitySSM:
    """
    Строит SSM из сегментного плана через матрицу музыкального сходства A.

    A[a, b] = сходство между барами с метками a и b.
    Матрица симметричная, диагональ = 1.0.

    Параметры постобработки (по умолчанию настроены на реалистичный режим):
      smooth_sigma  — Gaussian blur на границах блоков (σ=1.5 ≈ 1 такт).
      noise_std     — внутриблочный шум N(0, σ): имитирует вариацию chroma внутри секции.
      rescale       — если True, линейно растягивает значения под целевое распределение
                      chroma-SSM (target_mean=0.50, target_std=0.13).

    Обоснование:
      Модель обучена на chroma-SSM с косинусным сходством нормированных chroma-векторов.
      Типичное распределение: mean≈0.50, std≈0.13, плавные переходы.
      AffinitySSM без постобработки даёт ступенчатую матрицу с другим распределением,
      поэтому модель «не узнаёт» формат → внимание работает неоптимально.
      Постобработка приближает AffinitySSM к обучающему распределению без изменения
      блочной структуры (которая кодирует музыкальный план пользователя).
    """

    # Параметры целевого распределения, оценённые из датасета combined.
    # Chroma-SSM (cosine similarity нормированных chroma): mean≈0.50, std≈0.13.
    TARGET_MEAN = 0.50
    TARGET_STD  = 0.13

    def __init__(self, A: torch.Tensor, label_names: dict = None,
                 smooth_sigma: float = 1.5,
                 noise_std: float = 0.04,
                 rescale: bool = True):
        assert A.shape == (N_LABELS, N_LABELS), f"A должна быть {N_LABELS}×{N_LABELS}"
        self.A = A.float()
        self.label_names = label_names or LABEL_NAMES
        self.smooth_sigma = smooth_sigma
        self.noise_std = noise_std
        self.rescale = rescale

    @classmethod
    def fixed(cls, smooth_sigma: float = 1.5,
              noise_std: float = 0.04,
              rescale: bool = True) -> "AffinitySSM":
        """Создаёт AffinitySSM с теоретически заданной матрицей A."""
        A = torch.tensor(A_FIXED_VALUES, dtype=torch.float32)
        return cls(A, smooth_sigma=smooth_sigma, noise_std=noise_std, rescale=rescale)

    @classmethod
    def from_checkpoint(cls, path: str, smooth_sigma: float = 1.5,
                        noise_std: float = 0.04,
                        rescale: bool = True) -> "AffinitySSM":
        """Загружает эмпирически оценённую матрицу A из файла."""
        ckpt = torch.load(path, weights_only=True)
        return cls(ckpt["A"], label_names=ckpt.get("label_names", LABEL_NAMES),
                   smooth_sigma=smooth_sigma, noise_std=noise_std, rescale=rescale)

    def _build_bar_labels(self, segment_plan: List[Tuple[int, int]]) -> torch.Tensor:
        """
        segment_plan: [(label_id, duration_bars), ...]
        Возвращает: bar_labels [T]
        """
        T = sum(d for _, d in segment_plan)
        bar_labels = torch.full((T,), fill_value=7, dtype=torch.long)  # 7 = unknown
        cursor = 0
        for label_id, duration in segment_plan:
            bar_labels[cursor: cursor + duration] = label_id
            cursor += duration
        return bar_labels

    def build(self, segment_plan: List[Tuple[int, int]],
              ssm_size: int = 64) -> torch.Tensor:
        """
        Строит SSM [ssm_size, ssm_size] по сегментному плану.

        Постобработка (опционально):
          1. Gaussian smoothing — сглаживает резкие блочные границы.
          2. Внутриблочный шум — добавляет вариацию N(0, noise_std) внутри блоков,
             имитируя естественную вариабельность chroma-признаков внутри секции.
             Шум симметричен (верхний треугольник → нижний), диагональ = 1.0.
          3. Rescale — линейно растягивает значения под распределение реальной
             chroma-SSM (target_mean, target_std), сохраняя порядок значений.

        segment_plan: [(label_id, duration_bars), ...]
        """
        bar_labels = self._build_bar_labels(segment_plan)   # [T]
        ssm = self.A[bar_labels][:, bar_labels]             # [T, T]

        # 1. Gaussian smoothing
        if self.smooth_sigma > 0:
            ssm = _gaussian_blur_2d(ssm, self.smooth_sigma)

        # 2. Внутриблочный шум: симметричный, диагональ не трогаем
        if self.noise_std > 0:
            T = ssm.shape[0]
            noise = torch.randn(T, T) * self.noise_std
            # Симметризуем шум (SSM должна быть симметричной)
            noise = (noise + noise.T) / 2.0
            ssm = ssm + noise
            ssm.fill_diagonal_(1.0)
            ssm = ssm.clamp(0.0, 1.0)

        # 3. Rescale под целевое распределение chroma-SSM
        if self.rescale:
            # Работаем только с off-diagonal элементами
            diag_mask = ~torch.eye(ssm.shape[0], dtype=torch.bool)
            off = ssm[diag_mask]
            src_mean = off.mean()
            src_std  = off.std().clamp(min=1e-6)
            # Линейное масштабирование: z-score → целевое распределение
            ssm_rescaled = ssm.clone()
            ssm_rescaled[diag_mask] = (
                (off - src_mean) / src_std * self.TARGET_STD + self.TARGET_MEAN
            )
            ssm_rescaled[diag_mask] = ssm_rescaled[diag_mask].clamp(0.0, 1.0)
            ssm_rescaled.fill_diagonal_(1.0)
            ssm = ssm_rescaled

        return _resize_ssm(ssm, ssm_size)                   # [ssm_size, ssm_size]

    def summary(self) -> str:
        """Текстовый вывод матрицы A для отладки."""
        labels = [self.label_names.get(i, str(i)) for i in range(N_LABELS)]
        header = f"{'':>8}" + "".join(f"{l:>8}" for l in labels)
        lines = [header]
        for i, la in enumerate(labels):
            row = f"{la:>8}" + "".join(f"{self.A[i,j].item():>8.3f}"
                                        for j in range(N_LABELS))
            lines.append(row)
        return "\n".join(lines)


if __name__ == "__main__":
    # Быстрая проверка
    builder = AffinitySSM.fixed()
    print("=== A_fixed ===")
    print(builder.summary())

    plan = [(0, 8), (1, 16), (2, 16), (1, 16), (2, 16), (5, 8)]  # intro+verse+chorus...
    ssm = builder.build(plan, ssm_size=64)
    print(f"\n[realistic] SSM shape: {ssm.shape}")
    print(f"  mean={ssm.mean():.3f}, std={ssm.std():.3f}, "
          f"min={ssm.min():.3f}, max={ssm.max():.3f}")
    print(f"  (target: mean≈{AffinitySSM.TARGET_MEAN}, std≈{AffinitySSM.TARGET_STD})")

    ssm_raw = AffinitySSM.fixed(noise_std=0.0, rescale=False).build(plan, ssm_size=64)
    print(f"\n[raw]       SSM shape: {ssm_raw.shape}")
    print(f"  mean={ssm_raw.mean():.3f}, std={ssm_raw.std():.3f}, "
          f"min={ssm_raw.min():.3f}, max={ssm_raw.max():.3f}")
