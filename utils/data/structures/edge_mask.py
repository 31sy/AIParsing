import cv2
import os
import copy
import torch
import numpy as np

from models.ops.misc import interpolate

# transpose
FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class EdgeMask(object):
    """
    This class handles binary masks for all objects in the image
    """

    def __init__(self, edge_mask, size, mode=None):
        if isinstance(edge_mask, torch.Tensor):
            # The raw data representation is passed as argument
            edge_mask = edge_mask.clone()
        elif isinstance(edge_mask, (list, tuple)):
            edge_mask = torch.as_tensor(edge_mask)

        if len(edge_mask.shape) == 2:
            # if only a single instance mask is passed
            edge_mask = edge_mask[None]

        assert len(edge_mask.shape) == 3
        assert edge_mask.shape[1] == size[1], "%s != %s" % (edge_mask.shape[1], size[1])
        assert edge_mask.shape[2] == size[0], "%s != %s" % (edge_mask.shape[2], size[0])

        self.edge_mask = edge_mask
        self.size = size
        self.mode = mode
        self.extra_fields = {}

    def transpose(self, method):
        if method not in (FLIP_LEFT_RIGHT,):
            raise NotImplementedError("Only FLIP_LEFT_RIGHT implemented")

        dim = 1 if method == FLIP_TOP_BOTTOM else 2
        flipped_edge_mask = self.edge_mask.flip(dim)

        flipped_edge_mask = flipped_edge_mask.numpy()

        flipped_edge_mask = torch.from_numpy(flipped_edge_mask)

        return EdgeMask(flipped_edge_mask, self.size)

    def move(self, gap):
        c, h, w = self.edge_mask.shape
        old_up, old_left, old_bottom, old_right = max(gap[1], 0), max(gap[0], 0), h, w

        new_up, new_left = max(0 - gap[1], 0), max(0 - gap[0], 0)
        new_bottom, new_right = h + new_up - old_up, w + new_left - old_left
        new_shape = (c, h + new_up, w + new_left)

        moved_edge_mask = torch.zeros(new_shape, dtype=torch.uint8)
        moved_edge_mask[:, new_up:new_bottom, new_left:new_right] = \
            self.edge_mask[:, old_up:old_bottom, old_left:old_right]

        moved_size = new_shape[2], new_shape[1]
        return EdgeMask(moved_edge_mask, moved_size)

    def crop(self, box):
        assert isinstance(box, (list, tuple, torch.Tensor)), str(type(box))
        # box is assumed to be xyxy
        current_width, current_height = self.size
        xmin, ymin, xmax, ymax = [round(float(b)) for b in box]

        assert xmin <= xmax and ymin <= ymax, str(box)
        xmin = min(max(xmin, 0), current_width - 1)
        ymin = min(max(ymin, 0), current_height - 1)

        xmax = min(max(xmax, 0), current_width)
        ymax = min(max(ymax, 0), current_height)

        xmax = max(xmax, xmin + 1)
        ymax = max(ymax, ymin + 1)

        width, height = xmax - xmin, ymax - ymin
        cropped_edge_mask = self.edge_mask[:, ymin:ymax, xmin:xmax]
        cropped_size = width, height
        return EdgeMask(cropped_edge_mask, cropped_size)

    def set_size(self, size):
        c, h, w = self.edge_mask.shape
        new_shape = (c, size[1], size[0])

        new_edge_mask = torch.zeros(new_shape, dtype=torch.uint8)
        new_edge_mask[:, :min(h, size[1]), :min(w, size[0])] = \
            self.edge_mask[:, :min(h, size[1]), :min(w, size[0])]

        self.edge_mask = new_edge_mask
        return EdgeMask(self.edge_mask, size)

    def resize(self, size):
        try:
            iter(size)
        except TypeError:
            assert isinstance(size, (int, float))
            size = size, size
        width, height = map(int, size)

        assert width > 0
        assert height > 0

        # Height comes first here!
        resized_edge_mask = interpolate(
            self.edge_mask[None].float(),
            size=(height, width),
            mode="nearest",
        )[0].type_as(self.edge_mask)

        # resized_parsing = torch.from_numpy(
        #     cv2.resize(self.parsing.cpu().numpy().transpose(1, 2, 0),
        #     (width, height),
        #     interpolation=cv2.INTER_NEAREST).transpose(2, 0, 1)
        # ).type_as(self.parsing)

        resized_size = width, height
        return EdgeMask(resized_edge_mask, resized_size)

    def to(self, *args, **kwargs):
        return self

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def fields(self):
        return list(self.extra_fields.keys())

    def __len__(self):
        return len(self.edge_mask)

    def __getitem__(self, index):
        edge_mask = self.edge_mask[index].clone()
        return EdgeMask(edge_mask, self.size)

    def __iter__(self):
        self.iter_idx = 0
        return self

    def __next__(self):
        if self.iter_idx < self.__len__():
            next_edge_mask = self.__getitem__(self.iter_idx)
            self.iter_idx += 1
            return next_edge_mask
        raise StopIteration()

    def __repr__(self):
        s = self.__class__.__name__ + "("
        s += "num_edge_mask={}, ".format(len(self.edge_mask))
        s += "image_width={}, ".format(self.size[0])
        s += "image_height={}, ".format(self.size[1])
        return s


def get_edge_mask(root_dir, edge_mask_name):
    edge_mask_dir = root_dir.replace('img', 'edges')
    edge_mask_path = os.path.join(edge_mask_dir, edge_mask_name)
    return cv2.imread(edge_mask_path, 0)


def edge_mask_on_boxes(edge_mask, rois, heatmap_size):
    device = rois.device
    rois = rois.to(torch.device("cpu"))
    edge_mask_list = []
    for i in range(rois.shape[0]):
        edge_mask_ins = edge_mask[i].cpu().numpy()
        xmin, ymin, xmax, ymax = torch.round(rois[i]).int()
        cropped_edge_mask = edge_mask_ins[ymin:ymax, xmin:xmax]
        resized_edge_mask = cv2.resize(
            cropped_edge_mask,
            (heatmap_size[1], heatmap_size[0]),
            interpolation=cv2.INTER_NEAREST
        )
        edge_mask_list.append(torch.from_numpy(resized_edge_mask))

    if len(edge_mask_list) == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.stack(edge_mask_list, dim=0).to(device, dtype=torch.int64)


def flip_edge_mask_featuremap(edge_mask):
    edge_mask_flipped = edge_mask.copy()

    edge_mask_flipped = edge_mask_flipped[:, :, :, ::-1]

    return edge_mask_flipped
