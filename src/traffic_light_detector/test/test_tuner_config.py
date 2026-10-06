from traffic_light_detector.hsv_tuner_node import load_config, save_config


def test_gui_yaml_round_trip(tmp_path):
    target = tmp_path / "tuned.yaml"
    values = {"hsv.red1.h_min": 3, "hsv.red1.h_max": 11,
              "gui.config_save_path": str(target), "slot_boundaries": [0., .25, .5, .75, 1.]}
    save_config(str(target), values)
    loaded = load_config(str(target))
    assert loaded["hsv.red1.h_min"] == 3
    assert loaded["slot_boundaries"] == [0., .25, .5, .75, 1.]
