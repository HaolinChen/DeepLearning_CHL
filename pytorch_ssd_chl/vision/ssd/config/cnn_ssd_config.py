import numpy as np

from vision.utils.box_utils import cnnSSDSpec, SSDBoxSizes, generate_cnn_ssd_priors

image_size = [128, 160]
image_mean = np.array([127])  # GRAY layout
image_std = 128.0
image_norm = 256.0
iou_threshold = 0.1
center_variance = 0.1
size_variance = 0.2

specs = [
    cnnSSDSpec(16, 20, 8, 8, SSDBoxSizes(25.6, 44.8), [2, 3]),  # 0.2
    cnnSSDSpec(8, 10, 16, 16, SSDBoxSizes(44.8, 64), [2, 3]),  # 0.35
    cnnSSDSpec(4, 5, 32, 32, SSDBoxSizes(64, 83.2), [2, 3]),  # 0.5
    # cnnSSDSpec(2, 2, 64, 80, SSDBoxSizes(76.8, 102.4), [2, 3])  # 0.65
    cnnSSDSpec(2, 2, 64, 80, SSDBoxSizes(83.2, 102.4), [2, 3])  # 0.65
]


priors = generate_cnn_ssd_priors(specs, image_size)
