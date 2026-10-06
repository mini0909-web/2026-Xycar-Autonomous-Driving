from keyboard_drive.controller import (
    ControlConfig,
    KeyboardController,
    KEY_CENTER,
    KEY_DOWN,
    KEY_LEFT,
    KEY_QUIT,
    KEY_RIGHT,
    KEY_STOP,
    KEY_UP,
    consume_key_buffer,
)


def test_arrow_escape_sequences_and_partial_buffer():
    commands, remainder = consume_key_buffer('\x1b[A\x1b[D\x1b')
    assert commands == [KEY_UP, KEY_LEFT]
    assert remainder == '\x1b'

    commands, remainder = consume_key_buffer(remainder + '[C')
    assert commands == [KEY_RIGHT]
    assert remainder == ''


def test_speed_and_steering_are_clamped():
    controller = KeyboardController(
        ControlConfig(
            speed_step=2.0,
            steering_step=20.0,
            max_forward_speed=5.0,
            max_reverse_speed=3.0,
            max_steering=30.0,
        )
    )
    for index in range(5):
        controller.handle_key(KEY_UP, float(index))
        controller.handle_key(KEY_LEFT, float(index))
    assert controller.speed == 5.0
    assert controller.angle == -30.0

    for index in range(6):
        controller.handle_key(KEY_DOWN, 10.0 + index)
        controller.handle_key(KEY_RIGHT, 10.0 + index)
    assert controller.speed == -3.0
    assert controller.angle == 30.0


def test_stop_center_quit_and_deadman():
    controller = KeyboardController(ControlConfig(deadman_timeout_sec=0.5))
    controller.handle_key(KEY_UP, 1.0)
    controller.handle_key(KEY_LEFT, 1.0)
    assert not controller.apply_deadman(1.4)
    assert controller.apply_deadman(1.6)
    assert controller.speed == 0.0
    assert controller.angle == 0.0

    controller.handle_key(KEY_LEFT, 2.0)
    controller.handle_key(KEY_CENTER, 2.1)
    assert controller.angle == 0.0
    controller.handle_key(KEY_UP, 2.2)
    assert not controller.handle_key(KEY_STOP, 2.3)
    assert controller.speed == 0.0
    assert controller.handle_key(KEY_QUIT, 2.4)
