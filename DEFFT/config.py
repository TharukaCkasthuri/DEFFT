from dataclasses import dataclass

@dataclass
class FitnessCfg:
    clip_range: tuple[float, float] = (-0.9, 0.9)
    dp_sigma: float = 0.00
    reg_lambda: float = 0.01
    temperature: float = 0.5
    trim_fraction: float = 0.10
    score_is_loss: bool = False

