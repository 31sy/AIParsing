import os
import json
import pickle
import tempfile
import shutil
import numpy as np
from collections import OrderedDict

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import pdb
#from utils.data.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou

from utils.data.evaluation.densepose_cocoeval import denseposeCOCOeval
from utils.data.evaluation.parsing_eval import parsing_png, evaluate_parsing
from utils.misc import logging_rank
from maskrcnn_benchmark.data.datasets import dataset_catalog

from maskrcnn_benchmark.modeling.roi_heads.parsing_head.inference import parsing_results

from maskrcnn_benchmark.config import cfg
import cv2
import utils.colormap as colormap_utils
def post_processing_demo(results, imgs, filenames, output_folder):
    cpu_device = torch.device("cpu")
    results = [o.to(cpu_device) for o in results]
    num_im = len(imgs)

    ims_dets, ims_labels = prepare_box_results(results, imgs)


    par_results, par_score = prepare_parsing_results(results, imgs, filenames, output_folder)
    ims_pars = par_results

    return ims_dets, ims_labels, ims_pars


def prepare_box_results(results, imgs):
    box_results = []
    ims_dets = []
    ims_labels = []
    
    
    for i, result in enumerate(results):
        img = imgs[i]

        if len(result) == 0:
            ims_dets.append(None)
            ims_labels.append(None)
            continue
        
        
        image_height = img.shape[0]
        image_width = img.shape[1]
        #pdb.set_trace()
        result = result.resize((image_width, image_height))
        boxes = result.bbox
        scores = result.get_field("scores")
        labels = result.get_field("labels")
        ims_dets.append(np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False))
        result = result.convert("xywh")
        boxes = result.bbox.tolist()
        scores = scores.tolist()
        labels = labels.tolist()
        ims_labels.append(labels)


    return ims_dets, ims_labels


def prepare_parsing_results(results, imgs, img_names, output_folder):
    all_parsing = []
    all_scores = []
    #output_folder = os.path.join(cfg.OUTPUT_DIR, 'test')
    for i, result in enumerate(results):
        img = imgs[i]

        if len(result) == 0:
            ims_dets.append(None)
            ims_labels.append(None)
            continue
                
        image_height = img.shape[0]
        image_width = img.shape[1]
        
        result = result.resize((image_width, image_height))
        parsing = result.get_field("parsing")
        parsing = parsing_results(parsing, result, semseg=None)
        scores = result.get_field("parsing_scores")
        img_info={}
        img_info['height'] = image_height
        img_info['width'] = image_width
        img_info['file_name'] = img_names[i]
        parsing_png(
            parsing, scores, cfg.AIParsing.SEMSEG_SCORE_THRESH, img_info, output_folder, semseg=None
        )
        all_parsing.append(parsing)
        all_scores.append(scores)
    return all_parsing, all_scores


def parsing_png(parsing, scores, semseg_thresh, img_info, output_folder, semseg=None):
    parsing_output_dir = os.path.join(output_folder, 'parsing_predict')
    if not os.path.exists(parsing_output_dir):
        os.makedirs(parsing_output_dir)
    parsing_ins_dir = os.path.join(output_folder, 'parsing_instance')
    if not os.path.exists(parsing_ins_dir):
        os.makedirs(parsing_ins_dir)

    im_name = img_info['file_name']
    save_name_vis = os.path.join(parsing_output_dir, im_name.replace('.jpg', '_vis.png'))
    save_ins_vis = os.path.join(parsing_ins_dir, im_name.replace('.jpg', '_vis.png'))

    save_name = os.path.join(parsing_output_dir, im_name.replace('jpg', 'png'))
    save_ins = os.path.join(parsing_ins_dir, im_name.replace('jpg', 'png'))

    if semseg is not None:
        semseg = cv2.resize(semseg, (img_info["width"], img_info["height"]), interpolation=cv2.INTER_LINEAR)
        parsing_max = np.max(semseg, axis=2)
        max_map = np.where(parsing_max > 0.7, 1, 0)
        parsing_seg = np.argmax(semseg, axis=2).astype(np.uint8) * max_map
    else:
        parsing_seg = np.zeros((img_info["height"], img_info["width"]))
    parsing_ins = np.zeros((img_info["height"], img_info["width"]))

    _inx = scores.argsort()
    ins_id = 1
    for k in range(len(_inx)):
        if scores[_inx[k]] < semseg_thresh:
            continue
        _parsing = parsing[_inx[k]]
        parsing_seg = np.where(_parsing > 0, _parsing, parsing_seg)
        parsing_ins = np.where(_parsing > 0, ins_id, parsing_ins)
        ins_id += 1

    # get color map
    ins_colormap, parss_colormap = get_instance_parsing_colormap()

    # print(ins_colormap)
    # print(parss_colormap)
    import pdb
    #pdb.set_trace()
    parsing_seg_colormap = colormap_utils.dict2array(parss_colormap)
    parsing_ins_colormap = colormap_utils.dict2array(ins_colormap)

    cv2.imwrite(save_name, parsing_seg.astype(np.int))
    cv2.imwrite(save_ins, parsing_ins.astype(np.int))

    cv2.imwrite(save_name_vis, parsing_seg_colormap[parsing_seg.astype(np.int)])
    cv2.imwrite(save_ins_vis, parsing_ins_colormap[parsing_ins.astype(np.int)])

def get_box_result():
    box_results = []
    with open(dataset_catalog.get_ann_fn(cfg.TEST.DATASETS[0])) as f:
        anns = json.load(f)['annotations']
        for ann in anns:
            box_results.append({
                "image_id": ann['image_id'],
                "category_id": ann['category_id'],
                "bbox": ann['bbox'],
                "score": 1.0,
            })
            hier = ann['hier']
            N = len(hier) // 5
            for i in range(N):
                if hier[i * 5 + 4] > 0:
                    x1, y1, x2, y2 = hier[i * 5: i * 5 + 4]
                    bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
                    box_results.append({
                        "image_id": ann['image_id'],
                        "category_id": i + 2,
                        "bbox": bbox,
                        "score": 1.0,
                    })
    return box_results

def get_instance_parsing_colormap(rgb=False):
    instance_colormap = eval('colormap_utils.{}'.format(cfg.VIS.SHOW_BOX.COLORMAP))
    parsing_colormap = eval('colormap_utils.{}'.format(cfg.VIS.SHOW_PARSS.COLORMAP))
    if rgb:
        instance_colormap = colormap_utils.dict_bgr2rgb(instance_colormap)
        parsing_colormap = colormap_utils.dict_bgr2rgb(parsing_colormap)

    return instance_colormap, parsing_colormap


def vis_parsing(img, parsing, colormap, show_segms=True):
    """Visualizes a single binary parsing."""
    img = img.astype(np.float32)
    idx = np.nonzero(parsing)

    parsing_alpha = cfg.VIS.SHOW_PARSS.PARSING_ALPHA
    colormap = colormap_utils.dict2array(colormap)
    parsing_color = colormap[parsing.astype(np.int)]

    border_color = cfg.VIS.SHOW_PARSS.BORDER_COLOR
    border_thick = cfg.VIS.SHOW_PARSS.BORDER_THICK

    img[idx[0], idx[1], :] *= 1.0 - parsing_alpha
    # img[idx[0], idx[1], :] += alpha * parsing_color
    img += parsing_alpha * parsing_color

    if cfg.VIS.SHOW_PARSS.SHOW_BORDER and not show_segms:
        _, contours, _ = cv2.findContours(parsing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, border_color, border_thick, cv2.LINE_AA)

    return img.astype(np.uint8)

class COCOResults(object):
    METRICS = {
        "bbox": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "segm": ["AP", "AP50", "AP75", "APs", "APm", "APl"],
        "keypoints": ["AP", "AP50", "AP75", "APm", "APl"],
        "parsing": ["mIoU", "APp50", "APpvol", "PCP"],
        "uv": ["AP", "AP50", "AP75", "APm", "APl"],
    }

    def __init__(self, *iou_types):
        allowed_types = ("bbox", "segm", "keypoints", "parsing", "uv")
        assert all(iou_type in allowed_types for iou_type in iou_types)
        results = OrderedDict()
        for iou_type in iou_types:
            results[iou_type] = OrderedDict(
                [(metric, -1) for metric in COCOResults.METRICS[iou_type]]
            )
        self.results = results

    def update(self, coco_eval):
        if coco_eval is None:
            return

        assert isinstance(coco_eval, (COCOeval, denseposeCOCOeval))
        s = coco_eval.stats
        iou_type = coco_eval.params.iouType
        res = self.results[iou_type]
        metrics = COCOResults.METRICS[iou_type]
        if iou_type == 'uv':
            idx_map = [0, 1, 6, 11, 12]
            for idx, metric in enumerate(metrics):
                res[metric] = s[idx_map[idx]]
        else:
            for idx, metric in enumerate(metrics):
                res[metric] = s[idx]

    def update_parsing(self, eval):
        if eval is None:
            return

        res = self.results['parsing']
        for k, v in eval.items():
            res[k] = v
            
    def __repr__(self):
        # TODO make it pretty
        return repr(self.results)


def check_expected_results(results, expected_results, sigma_tol):
    if not expected_results:
        return

    logger = logging.getLogger("inference")
    for task, metric, (mean, std) in expected_results:
        actual_val = results.results[task][metric]
        lo = mean - sigma_tol * std
        hi = mean + sigma_tol * std
        ok = (lo < actual_val) and (actual_val < hi)
        msg = (
            "{} > {} sanity check (actual vs. expected): "
            "{:.3f} vs. mean={:.4f}, std={:.4}, range=({:.4f}, {:.4f})"
        ).format(task, metric, actual_val, mean, std, lo, hi)
        if not ok:
            msg = "FAIL: " + msg
            logger.error(msg)
        else:
            msg = "PASS: " + msg
            logger.info(msg)
