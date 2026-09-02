"""Deterministic planar UAV dynamics shared by all navigation tasks.

The paper models the UAV state as ``[x, y, v, theta]`` and applies an
instantaneous heading increment and/or acceleration before translating the
vehicle.  This module keeps that model independent from Gymnasium and from any
learning algorithm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


MOVE = 0
TURN = 1
CATCH = 2


@dataclass(frozen=True)
class DynamicsConfig:
    """Physical scaling used to turn normalized parameters into controls."""

    max_acceleration: float = 4.0
    max_turn_angle: float = math.pi / 3.0
    min_speed: float = 0.0
    max_speed: float = 40.0
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.max_acceleration <= 0 or self.max_turn_angle <= 0:
            raise ValueError("Control limits must be positive")
        if self.dt <= 0 or self.min_speed < 0 or self.max_speed <= self.min_speed:
            raise ValueError("Invalid time step or speed limits")


@dataclass
class UAVState:
    """Planar UAV state in metres, metres/step, and radians."""

    x: float
    y: float
    speed: float
    heading: float

    @property
    def position(self) -> np.ndarray:
        return np.asarray([self.x, self.y], dtype=np.float32)

    def copy(self) -> "UAVState":
        return UAVState(self.x, self.y, self.speed, self.heading)


def wrap_angle(angle: float) -> float:
    """Wrap an angle to ``[-pi, pi)``."""

    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def clip_normalized_parameter(parameter: float | np.ndarray) -> float:
    """Convert the action parameter to a scalar and clip it to ``[-1, 1]``."""

    values = np.asarray(parameter, dtype=np.float32).reshape(-1)
    if values.size != 1:
        raise ValueError("Each benchmark action has exactly one parameter slot")
    return float(np.clip(values[0], -1.0, 1.0))


def advance(
    state: UAVState,
    discrete_action: int,
    normalized_parameter: float | np.ndarray,
    config: DynamicsConfig,
) -> UAVState:
    """Apply one paper-style dynamics step.

    ``MOVE`` changes speed, ``TURN`` changes heading, and ``CATCH`` changes
    neither.  Every action then advances the UAV using its resulting speed and
    heading, matching Eq. (5) and the paper's statement that an unsuccessful
    CATCH advances one step.
    """

    parameter = clip_normalized_parameter(normalized_parameter)
    speed = float(state.speed)
    heading = float(state.heading)

    if discrete_action == MOVE:
        speed += parameter * config.max_acceleration * config.dt
    elif discrete_action == TURN:
        heading += parameter * config.max_turn_angle
    elif discrete_action != CATCH:
        raise ValueError(f"Unknown discrete action: {discrete_action}")

    speed = float(np.clip(speed, config.min_speed, config.max_speed))
    heading = wrap_angle(heading)
    x = state.x + speed * math.cos(heading) * config.dt
    y = state.y + speed * math.sin(heading) * config.dt
    return UAVState(float(x), float(y), speed, heading)


def distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Euclidean distance supporting both scalar and batched goal arrays."""

    first_array = np.asarray(first, dtype=np.float32)
    second_array = np.asarray(second, dtype=np.float32)
    return np.linalg.norm(first_array - second_array, axis=-1)
