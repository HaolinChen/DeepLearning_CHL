import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 240
image_mean = np.array([127])  # GRAY layout
image_std = 128.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(30, 8, SSDBoxSizes(24, 72), [2]),
    SSDSpec(15, 16, SSDBoxSizes(72, 120), [2, 3]),
    SSDSpec(7, 32, SSDBoxSizes(120, 168), [2, 3]),
    SSDSpec(3, 80, SSDBoxSizes(168, 216), [2, 3]),
    SSDSpec(1, 240, SSDBoxSizes(216, 264), [2]),
]


priors = generate_ssd_priors(specs, image_size)
