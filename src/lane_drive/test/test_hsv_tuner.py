from pathlib import Path

from lane_drive.hsv_tuner_node import load_lane_parameters, update_yaml_values


def test_updates_only_selected_hsv_values_and_preserves_comments(tmp_path: Path):
    config = tmp_path / 'lane_drive.yaml'
    config.write_text(
        'lane_drive:\n'
        '  ros__parameters:\n'
        '    max_speed: 10.0\n'
        '    white_min_saturation: 70 # max S\n'
        '    white_min_value: 175\n'
        '    yellow_hue_min: 15\n'
        '    yellow_hue_max: 40\n'
        '    yellow_min_saturation: 70\n'
        '    yellow_min_value: 80\n',
        encoding='utf-8',
    )
    values = {
        'white_min_saturation': 42,
        'white_min_value': 190,
        'yellow_hue_min': 18,
        'yellow_hue_max': 36,
        'yellow_min_saturation': 90,
        'yellow_min_value': 95,
    }

    update_yaml_values(config, values)

    text = config.read_text(encoding='utf-8')
    assert 'max_speed: 10.0' in text
    assert 'white_min_saturation: 42 # max S' in text
    assert load_lane_parameters(config)['yellow_min_value'] == 95
