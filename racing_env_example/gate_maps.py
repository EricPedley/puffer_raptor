import numpy as np


gate_height = -1.35 - 0.1
# DEFINE RACE TRACK
positions_with_extr_gate = np.array(
    [
        [-13.5, 6.0, gate_height],
        [-11.0, 14.0, gate_height],
        [-6.0, 22.0, gate_height],
        [-11.0, 30.0, gate_height],
        [-11.0 - 2.1 * np.cos(np.pi / 3.0), 30.0 + 2.1 * np.sin(np.pi / 3.0), -5.1],
        [-11.0, 30.0, gate_height - 2.7],
        [-19.0, 34.0, gate_height],
        [-27.0, 30.0, gate_height],
        [-32.0, 22.0, gate_height],
        [-29.0, 14.0, gate_height],
        [-30.0, 6.0, gate_height],
        [-17.0, 18.0, gate_height],
        [-13.5, 6.0, gate_height - 2.7],
    ]
)
yaws_with_extra_gate = (
    np.array(
        [
            7 / 12,
            1 / 3,
            2 / 3,
            1 / 6,
            -5 / 6,
            1 / 6,
            0,
            -1 / 6,
            -1 / 2,
            -5 / 12,
            -7 / 12,
            1,
            -5 / 12,
        ]
    )
    * np.pi
)
easy_start = positions_with_extr_gate[0] + np.array([1.0, -3.0, 0])

gate_positions = np.delete(positions_with_extr_gate, 4, axis=0)
gate_yaws = np.delete(yaws_with_extra_gate, 4)
racetrack_start = gate_positions[0] + np.array([1.0, -3.0, 1.45])
