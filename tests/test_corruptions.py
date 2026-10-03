import numpy as np
from src.data.corruptions import *

IMG = (np.random.RandomState(0).rand(128, 128, 3) * 200 + 20).astype(np.uint8)


def test_deterministic():
    rng = np.random.default_rng(1)
    for cond in fixed_test_conditions(rng):
        assert np.array_equal(apply_condition(IMG, cond), apply_condition(IMG, cond))


def test_salt_pepper_rate_and_values():
    out = apply_condition(IMG, {"type": "salt_pepper", "p": 0.15, "seed": 3})
    changed = (out != IMG).any(-1)
    assert abs(changed.mean() - 0.15) < 0.02
    assert set(np.unique(out[changed])) <= {0, 255}


def test_blur_smooths_image():
    out = apply_condition(IMG, {"type": "blur", "kernel": 7, "sigma": 2.5, "seed": 0})
    assert out.std() < IMG.std()


def test_occlusion_coverage_levels():
    rng = np.random.default_rng(0)
    for n, cov in TEST_OCC:
        cs = [occlusion_coverage(sample_boxes(rng, n, cov)) for _ in range(300)]
        assert abs(np.mean(cs) - cov) < 0.02, (n, cov, np.mean(cs))


def test_train_sampler_equal_class_probability_and_ranges():
    rng = np.random.default_rng(0)
    conds = [sample_train_condition(rng) for _ in range(8000)]
    freq = np.bincount([label_of(c) for c in conds], minlength=4) / len(conds)
    assert np.allclose(freq, 0.25, atol=0.02)
    for c in conds:
        if c["type"] == "salt_pepper": assert 0.02 <= c["p"] <= 0.15
        if c["type"] == "blur": assert c["kernel"] in (3, 5, 7) and 0.5 <= c["sigma"] <= 2.5
        if c["type"] == "occlusion": assert 1 <= len(c["boxes"]) <= 3


def test_fixed_test_has_10_conditions():
    assert len(fixed_test_conditions(np.random.default_rng(0))) == 10


def test_sample_condition_respects_type_and_ranges():
    rng = np.random.default_rng(0)
    for t in CLASSES:
        for _ in range(200):
            c = sample_condition(rng, t)
            assert c["type"] == t
            if t == "salt_pepper": assert 0.02 <= c["p"] <= 0.15
            if t == "blur": assert c["kernel"] in (3, 5, 7)
            if t == "occlusion": assert 1 <= len(c["boxes"]) <= 3