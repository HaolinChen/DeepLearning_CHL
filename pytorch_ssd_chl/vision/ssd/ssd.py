import torch.nn as nn
import torch
import numpy as np
from typing import List, Tuple
import torch.nn.functional as F

from ..utils import box_utils
from collections import namedtuple

GraphPath = namedtuple("GraphPath", ['s0', 'name', 's1'])  #

torch.set_printoptions(threshold=np.inf)


class SSD(nn.Module):
    def __init__(self, num_classes: int, base_net: nn.ModuleList, source_layer_indexes: List[int],
                 extras: nn.ModuleList, classification_headers: nn.ModuleList,
                 regression_headers: nn.ModuleList, is_test=False, config=None, device=None):
        """Compose a SSD model using the given components.
        """
        super(SSD, self).__init__()

        self.num_classes = num_classes
        self.base_net = base_net
        self.source_layer_indexes = source_layer_indexes
        self.extras = extras
        self.classification_headers = classification_headers
        self.regression_headers = regression_headers
        self.is_test = is_test
        self.config = config

        # register layers in source_layer_indexes by adding them to a module list
        self.source_layer_add_ons = nn.ModuleList([t[1] for t in source_layer_indexes
                                                   if isinstance(t, tuple) and not isinstance(t, GraphPath)])
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if is_test:
            self.config = config
            self.priors = config.priors.to(self.device)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        confidences = []
        locations = []
        start_layer_index = 0
        header_index = 0
        for end_layer_index in self.source_layer_indexes:
            if isinstance(end_layer_index, GraphPath):
                path = end_layer_index
                end_layer_index = end_layer_index.s0
                added_layer = None
            elif isinstance(end_layer_index, tuple):
                added_layer = end_layer_index[1]
                end_layer_index = end_layer_index[0]
                path = None
            else:
                added_layer = None
                path = None
            for layer in self.base_net[start_layer_index: end_layer_index]:
                x = layer(x)
            if added_layer:
                y = added_layer(x)
            else:
                y = x
            if path:
                sub = getattr(self.base_net[end_layer_index], path.name)
                for layer in sub[:path.s1]:
                    x = layer(x)
                y = x
                for layer in sub[path.s1:]:
                    x = layer(x)
                end_layer_index += 1
            start_layer_index = end_layer_index
            confidence, location = self.compute_header(header_index, y)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        for layer in self.base_net[end_layer_index:]:
            x = layer(x)

        for layer in self.extras:
            x = layer(x)
            confidence, location = self.compute_header(header_index, x)
            header_index += 1
            confidences.append(confidence)
            locations.append(location)

        confidences = torch.cat(confidences, 1)
        locations = torch.cat(locations, 1)

        if self.is_test:
            confidences = F.softmax(confidences, dim=2)
            boxes = box_utils.convert_locations_to_boxes(
                locations, self.priors, self.config.center_variance, self.config.size_variance
            )
            # with open('out_debug/output56_confidence.txt', "w") as f:
            #     confidences_out6 = confidences[:, 16 * 20 * 6 + 8 * 10 * 6:16 * 20 * 6 + 8 * 10 * 6 + 4 * 5 * 6, :]
            #     boxes_out6 = boxes[:, 16 * 20 * 6 + 8 * 10 * 6:16 * 20 * 6 + 8 * 10 * 6 + 4 * 5 * 6, :]
            #     locations_out6 = locations[:, 16 * 20 * 6 + 8 * 10 * 6:16 * 20 * 6 + 8 * 10 * 6 + 4 * 5 * 6, :]
            #     for i in range(0, 4 * 5 * 6):
            #         if confidences_out6[:, i, 1] > 0.3:
            #             print(confidences_out6[:, i, :], file=f)
            #             print(boxes_out6[:, i, :], file=f)
            #             print(locations_out6[:, i, :], file=f)
            #             print(i, file=f)

            boxes = box_utils.center_form_to_corner_form(boxes)
            return confidences, boxes
        else:
            return confidences, locations
        # return confidences[0], locations[0], confidences[1], locations[1], confidences[2], locations[2], confidences[3], \
        #        locations[3]

    def compute_header(self, i, x):
        confidence = self.classification_headers[i](x)
        confidence = confidence.permute(0, 2, 3, 1).contiguous()
        confidence = confidence.view(confidence.size(0), -1, self.num_classes)

        location = self.regression_headers[i](x)
        location = location.permute(0, 2, 3, 1).contiguous()  # ?????????????????? batch*channel*size*size -> batch*size*size*24
        # if i == 2:
        #     with open('out_debug/output5_permute.txt', "w") as f:
        #         # torch.set_printoptions(threshold=np.inf)
        #         print(location, file=f)
        location = location.view(location.size(0), -1, 4)  # ?????????????????? batch*size*size*channel -> batch*box_num*4
        # if i == 2:
        #     with open('out_debug/output5_view.txt', "w") as f:
        #         # torch.set_printoptions(threshold=np.inf)
        #         print(location, file=f)
        return confidence, location

    def init_from_base_net(self, model):
        self.base_net.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage), strict=True)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init_from_pretrained_ssd(self, model):
        state_dict = torch.load(model, map_location=lambda storage, loc: storage)
        state_dict = {k: v for k, v in state_dict.items() if
                      not (k.startswith("classification_headers") or k.startswith("regression_headers"))}
        model_dict = self.state_dict()
        model_dict.update(state_dict)
        self.load_state_dict(model_dict)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def init(self):
        self.base_net.apply(_xavier_init_)
        self.source_layer_add_ons.apply(_xavier_init_)
        self.extras.apply(_xavier_init_)
        self.classification_headers.apply(_xavier_init_)
        self.regression_headers.apply(_xavier_init_)

    def load(self, model):
        self.load_state_dict(torch.load(model, map_location=lambda storage, loc: storage))

    def save(self, model_path):
        torch.save(self.state_dict(), model_path)


class MatchPrior(object):
    def __init__(self, center_form_priors, center_variance, size_variance, iou_threshold):
        self.center_form_priors = center_form_priors
        self.corner_form_priors = box_utils.center_form_to_corner_form(center_form_priors)
        self.center_variance = center_variance
        self.size_variance = size_variance
        self.iou_threshold = iou_threshold

    def __call__(self, gt_boxes, gt_labels):
        if type(gt_boxes) is np.ndarray:
            gt_boxes = torch.from_numpy(gt_boxes)
        if type(gt_labels) is np.ndarray:
            gt_labels = torch.from_numpy(gt_labels)
        boxes, labels = box_utils.assign_priors(gt_boxes, gt_labels,
                                                self.corner_form_priors, self.iou_threshold)
        boxes = box_utils.corner_form_to_center_form(boxes)
        locations = box_utils.convert_boxes_to_locations(boxes, self.center_form_priors, self.center_variance,
                                                         self.size_variance)
        return locations, labels


def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)
