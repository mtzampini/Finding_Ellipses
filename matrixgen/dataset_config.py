from dataclasses import dataclass, field
from typing import Dict


@dataclass
class DatasetConfig:
    n_total_core: int = 4000

    case_proportions: Dict[int, float] = field(default_factory=lambda: {
        1: 0.15,   # normal
        2: 0.25,   # reducible
        3: 0.10,   # flat boundary 
        4: 0.50,   # generic
    })

    case2_inside_fraction = 0.5

    # augmentation
    n_augment_per_case: Dict[int, int] = field(default_factory=lambda: {
        1: 5,
        2: 5,
        3: 8,   
        4: 5,
    })

    n_hard_per_case: int = 200

    train_fraction: float = 0.8

    seed: int = 42

    generator_kwargs = {
    1: {
        "distinct_eigvals": 3,
    },

    2: {
        "scale": 2.0,
    },

    3: {
        "scale": 2.0,
        "n_theta": 720,
        "flat_tol": 2e-3,
    },

    4: {
        "scale": 2.0,
        "n_theta": 720,
        "flat_tol": 1e-3,
        "max_attempts": 100,
    },
    }

    def __post_init__(self):
        total = sum(self.case_proportions.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"case_proportions sum must be 1.0, found {total}")
        if not (0 < self.train_fraction < 1):
            raise ValueError("train_fraction must be between 0 and 1")

    def n_per_case(self) -> Dict[int, int]:
        """Number of matrices (pre-augmentation) to generate for each case"""
        return {
            case: round(self.n_total_core * prop)
            for case, prop in self.case_proportions.items()
        }


if __name__ == "__main__":
    cfg = DatasetConfig()
    print(cfg.n_per_case())