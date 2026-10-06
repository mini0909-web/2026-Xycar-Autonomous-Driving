"""Pure key parsing and command-state logic for keyboard driving."""

from dataclasses import dataclass
from typing import List, Optional, Tuple


KEY_UP = 'UP'
KEY_DOWN = 'DOWN'
KEY_LEFT = 'LEFT'
KEY_RIGHT = 'RIGHT'
KEY_CENTER = 'CENTER'
KEY_STOP = 'STOP'
KEY_QUIT = 'QUIT'


ESCAPE_KEYS = {
    '\x1b[A': KEY_UP,
    '\x1b[B': KEY_DOWN,
    '\x1b[C': KEY_RIGHT,
    '\x1b[D': KEY_LEFT,
}

CHARACTER_KEYS = {
    'c': KEY_CENTER,
    'C': KEY_CENTER,
    ' ': KEY_STOP,
    'q': KEY_QUIT,
    'Q': KEY_QUIT,
}


def consume_key_buffer(buffer: str) -> Tuple[List[str], str]:
    """Extract complete arrow/character commands and retain a partial escape."""
    commands: List[str] = []
    index = 0
    while index < len(buffer):
        if buffer[index] == '\x1b':
            remaining = len(buffer) - index
            if remaining < 3:
                return commands, buffer[index:]
            sequence = buffer[index:index + 3]
            command = ESCAPE_KEYS.get(sequence)
            if command is not None:
                commands.append(command)
                index += 3
            else:
                index += 1
            continue

        command = CHARACTER_KEYS.get(buffer[index])
        if command is not None:
            commands.append(command)
        index += 1
    return commands, ''


@dataclass(frozen=True)
class ControlConfig:
    """Limits and increments for keyboard control."""

    speed_step: float = 1.0
    steering_step: float = 5.0
    max_forward_speed: float = 6.0
    max_reverse_speed: float = 3.0
    max_steering: float = 50.0
    left_steering_sign: float = -1.0
    deadman_timeout_sec: float = 0.5


class KeyboardController:
    """Maintain bounded angle/speed and apply a no-input deadman stop."""

    def __init__(self, config: Optional[ControlConfig] = None):
        self.config = config or ControlConfig()
        self.angle = 0.0
        self.speed = 0.0
        self.last_command_time: Optional[float] = None

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(maximum, value))

    def handle_key(self, key: str, now: float) -> bool:
        """Apply one command. Return True when the user requested exit."""
        config = self.config
        if key == KEY_UP:
            self.speed = self._clamp(
                self.speed + config.speed_step,
                -abs(config.max_reverse_speed),
                abs(config.max_forward_speed),
            )
        elif key == KEY_DOWN:
            self.speed = self._clamp(
                self.speed - config.speed_step,
                -abs(config.max_reverse_speed),
                abs(config.max_forward_speed),
            )
        elif key == KEY_LEFT:
            direction = -1.0 if config.left_steering_sign < 0.0 else 1.0
            self.angle = self._clamp(
                self.angle + direction * config.steering_step,
                -abs(config.max_steering),
                abs(config.max_steering),
            )
        elif key == KEY_RIGHT:
            direction = 1.0 if config.left_steering_sign < 0.0 else -1.0
            self.angle = self._clamp(
                self.angle + direction * config.steering_step,
                -abs(config.max_steering),
                abs(config.max_steering),
            )
        elif key == KEY_CENTER:
            self.angle = 0.0
        elif key in (KEY_STOP, KEY_QUIT):
            self.angle = 0.0
            self.speed = 0.0
        else:
            return False

        self.last_command_time = now
        return key == KEY_QUIT

    def apply_deadman(self, now: float) -> bool:
        """Stop and center if motion commands have gone stale."""
        timeout = float(self.config.deadman_timeout_sec)
        if timeout <= 0.0 or self.last_command_time is None:
            return False
        if now - self.last_command_time <= timeout:
            return False
        if self.speed == 0.0 and self.angle == 0.0:
            return False
        self.speed = 0.0
        self.angle = 0.0
        return True
