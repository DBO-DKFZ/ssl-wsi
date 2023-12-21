import numpy as np


def cosine_scheduler(
    start_value: float,
    final_value: float,
    total_steps: int,
    warmup_steps: int = 0,
    warmup_value: float = 1e-6,
):
    warmup_schedule = np.array([])
    if warmup_steps > 0:
        warmup_schedule = np.linspace(warmup_value, start_value, warmup_steps)

    steps = np.arange(total_steps - warmup_steps)
    schedule = final_value + 0.5 * (start_value - final_value) * (
        1 + np.cos(np.pi * steps / len(steps))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == total_steps
    return schedule
