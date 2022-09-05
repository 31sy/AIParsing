import cv2
import numpy as np
import pycocotools.mask as mask_util
import random

import pdb
import torch

#from utils.data.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from utils.data.structures.boxlist_ops import cat_boxlist, boxlist_nms, \
    boxlist_ml_nms, boxlist_soft_nms, boxlist_box_voting
from utils.data.structures.parsing import flip_parsing_featuremap
from utils.data.structures.densepose_uv import flip_uv_featuremap
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.transforms import transforms as T



def im_detect_bbox(model, ims):
    box_results = [[] for _ in range(len(ims))]
    features = []
    results, net_imgs_size, blob_conv = im_detect_bbox_net(model, ims, cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MAX_SIZE_TEST)

    add_results(box_results, results)
    features.append((net_imgs_size, blob_conv))


    box_results = [cat_boxlist(result) for result in box_results]

    if cfg.MODEL.FASTER_ON:
        box_results = [filter_results(result) for result in box_results]

    return box_results, features




def im_detect_parsing(model, rois, features):
    _idx = 0
    parsing_results = [[] for _ in range(len(rois))]
    parsing_scores = [[] for _ in range(len(rois))]
    conv_features = features[_idx][1]
    _idx += 1
    results = model.parsing_net(conv_features, rois, targets=None)

    return results





def im_detect_bbox_net(model, ims, target_scale, target_max_size, flip=False, size=None, target=None):
    net_imgs_size = []
    results = []
    ims_blob = get_blob(ims, target_scale, target_max_size, flip)
    import pdb
    #pdb.set_trace()
    blob_conv, _results = model.box_net(ims_blob,target)

    for i, im_result in enumerate(_results):
        net_img_size = im_result.size
        net_imgs_size.append(net_img_size)
        if flip:
            im_result = im_result.transpose(0)
            if len(cfg.TRAIN.LEFT_RIGHT) > 0:
                scores = im_result.get_field("scores").reshape(-1, cfg.MODEL.NUM_CLASSES)
                boxes = im_result.bbox.reshape(-1, cfg.MODEL.NUM_CLASSES, 4)
                idx = torch.arange(cfg.MODEL.NUM_CLASSES)
                for j in cfg.TRAIN.LEFT_RIGHT:
                    idx[j[0]] = j[1]
                    idx[j[1]] = j[0]
                boxes = boxes[:, idx].reshape(-1, 4)
                scores = scores[:, idx].reshape(-1)
                im_result.bbox = boxes
                im_result.add_field("scores", scores)
        if size:
            im_result = im_result.resize(size[i])
        results.append(im_result)

    return results, net_imgs_size, blob_conv


def add_results(all_results, results):
    for i in range(len(all_results)):
        all_results[i].append(results[i])



def get_size(image_size,min_size,max_size):
    #pdb.set_trace()
    w, h = image_size[1],image_size[0]
    size = min_size
    max_size = max_size
    if max_size is not None:
        min_original_size = float(min((w, h)))
        max_original_size = float(max((w, h)))
        if max_original_size / min_original_size * size > max_size:
            size = int(round(max_size * min_original_size / max_original_size))

    if (w <= h and w == size) or (h <= w and h == size):
        return (h, w)

    if w < h:
        ow = size
        oh = int(size * h / w)
    else:
        oh = size
        ow = int(size * w / h)

    return (oh, ow)
def get_blob(ims, target_scale, target_max_size, flip):
    
    ims_processed = []
    for im in ims:
        if flip:
            im = im[:, ::-1, :]
        #import pdb
        #
        im = im.astype(np.float32, copy=False)
        #im -= cfg.INPUT.PIXEL_MEAN
        im_shape = im.shape #[h,w,3]
        #new_size = get_size(im_shape,target_scale,target_max_size)
        #im_resized = cv2.resize(im, new_size, interpolation=cv2.INTER_LINEAR)
        #pdb.set_trace()
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])

        im_scale = float(target_scale) / float(im_size_min)
        # Prevent the biggest axis from being more than max_size
        if np.round(im_scale * im_size_max) > target_max_size:
            im_scale = float(target_max_size) / float(im_size_max)
        
        im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
        im_resized -= cfg.INPUT.PIXEL_MEAN
        im_processed = im_resized.transpose(2, 0, 1)
        im_processed = torch.from_numpy(im_processed).to(torch.device(cfg.MODEL.DEVICE))
        #print(im_processed)
        ims_processed.append(im_processed)

    return ims_processed


def filter_results(boxlist):
    num_classes = 1 #cfg.MODEL.NUM_CLASSES
    import pdb
    #pdb.set_trace()
    if not cfg.TEST.SOFT_NMS.ENABLED and not cfg.TEST.BBOX_VOTE.ENABLED:
        # multiclass nms
        scores = boxlist.get_field("scores")
        device = scores.device
        num_repeat = int(boxlist.bbox.shape[0] / num_classes)
        labels = np.tile(np.arange(num_classes), num_repeat)
        boxlist.add_field("labels", torch.from_numpy(labels).to(dtype=torch.int64, device=device))
        fg_labels = torch.from_numpy(
            (np.arange(boxlist.bbox.shape[0]) % num_classes != 0).astype(int)
        ).to(dtype=torch.uint8, device=device)
        _scores = scores > cfg.FAST_RCNN.SCORE_THRESH
        inds_all = _scores & fg_labels
        result = boxlist_ml_nms(boxlist[inds_all], cfg.FAST_RCNN.NMS)
    else:
        #pdb.set_trace()
        boxes = boxlist.bbox.reshape(-1, num_classes * 4)
        scores = boxlist.get_field("scores").reshape(-1, num_classes)
        device = scores.device
        result = []
        # Apply threshold on detection probabilities and apply NMS
        # Skip j = 0, because it's the background class
        inds_all = scores > cfg.FAST_RCNN.SCORE_THRESH
        for j in range(0, num_classes):
            inds = inds_all[:, j].nonzero().squeeze(1)
            scores_j = scores[inds, j]
            boxes_j = boxes[inds, j * 4: (j + 1) * 4]
            boxlist_for_class = BoxList(boxes_j, boxlist.size, mode="xyxy")
            boxlist_for_class.add_field("scores", scores_j)
            boxlist_for_class_old = boxlist_for_class
            if cfg.TEST.SOFT_NMS.ENABLED:
                boxlist_for_class = boxlist_soft_nms(
                    boxlist_for_class,
                    sigma=cfg.TEST.SOFT_NMS.SIGMA,
                    overlap_thresh=cfg.FAST_RCNN.NMS,
                    score_thresh=0.0001,
                    method=cfg.TEST.SOFT_NMS.METHOD
                )
            else:
                boxlist_for_class = boxlist_nms(
                    boxlist_for_class, cfg.FAST_RCNN.NMS
                )
            # Refine the post-NMS boxes using bounding-box voting
            if cfg.TEST.BBOX_VOTE.ENABLED and boxes_j.shape[0] > 0:
                boxlist_for_class = boxlist_box_voting(
                    boxlist_for_class,
                    boxlist_for_class_old,
                    cfg.TEST.BBOX_VOTE.VOTE_TH,
                    scoring_method=cfg.TEST.BBOX_VOTE.SCORING_METHOD
                )
            num_labels = len(boxlist_for_class)
            boxlist_for_class.add_field(
                "labels", torch.full((num_labels,), j+1, dtype=torch.int64, device=device)
            )
            result.append(boxlist_for_class)

        result = cat_boxlist(result)


    number_of_detections = len(result)

    # Limit to max_per_image detections **over all classes**
    if number_of_detections > cfg.FAST_RCNN.DETECTIONS_PER_IMG > 0:
        cls_scores = result.get_field("scores")
        image_thresh, _ = torch.kthvalue(
            cls_scores.cpu(), number_of_detections - cfg.FAST_RCNN.DETECTIONS_PER_IMG + 1
        )
        keep = cls_scores >= image_thresh.item()
        keep = torch.nonzero(keep).squeeze(1)
        result = result[keep]
    return result


