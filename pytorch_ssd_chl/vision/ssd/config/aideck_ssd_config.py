import numpy as np

from vision.utils.box_utils import SSDSpec, SSDBoxSizes, generate_ssd_priors

image_size = 120
image_mean = np.array([127])  # GRAY layout
image_std = 128.0
image_norm = 256.0
iou_threshold = 0.45
center_variance = 0.1
size_variance = 0.2

specs = [
    SSDSpec(60, 2, SSDBoxSizes(12, 36), [2, 3]),
    SSDSpec(30, 4, SSDBoxSizes(36, 60), [2, 3]),
    SSDSpec(15, 8, SSDBoxSizes(60, 84), [2, 3]),
    SSDSpec(7, 18, SSDBoxSizes(84, 108), [2, 3]),
]

priors = generate_ssd_priors(specs, image_size)
