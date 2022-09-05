import cv2
import numpy as np

import torch
from torch.nn import functional as F

from utils.data.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.modeling.matcher import Matcher
from maskrcnn_benchmark.utils.misc import cat, keep_only_positive_boxes, across_sample
from maskrcnn_benchmark.config import cfg
import pdb
import maskrcnn_benchmark.layers.lovasz_losses as L


def generate_edge_tensor(label, edge_width=3):
    label = label.type(torch.cuda.FloatTensor)
    if len(label.shape) == 2:
        label = label.unsqueeze(0)
    n, h, w = label.shape
    edge = torch.zeros(label.shape, dtype=torch.float).cuda()
    # right
    edge_right = edge[:, 1:h, :]
    edge_right[(label[:, 1:h, :] != label[:, :h - 1, :]) & (label[:, 1:h, :] != 255)
               & (label[:, :h - 1, :] != 255)] = 1

    # up
    edge_up = edge[:, :, :w - 1]
    edge_up[(label[:, :, :w - 1] != label[:, :, 1:w])
            & (label[:, :, :w - 1] != 255)
            & (label[:, :, 1:w] != 255)] = 1

    # upright
    edge_upright = edge[:, :h - 1, :w - 1]
    edge_upright[(label[:, :h - 1, :w - 1] != label[:, 1:h, 1:w])
                 & (label[:, :h - 1, :w - 1] != 255)
                 & (label[:, 1:h, 1:w] != 255)] = 1

    # bottomright
    edge_bottomright = edge[:, :h - 1, 1:w]
    edge_bottomright[(label[:, :h - 1, 1:w] != label[:, 1:h, :w - 1])
                     & (label[:, :h - 1, 1:w] != 255)
                     & (label[:, 1:h, :w - 1] != 255)] = 1

    kernel = torch.ones((1, 1, edge_width, edge_width), dtype=torch.float).cuda()
    with torch.no_grad():
        edge = edge.unsqueeze(1)
        edge = F.conv2d(edge, kernel, stride=1, padding=1)
    edge[edge!=0] = 1
    edge = edge.squeeze()
    return edge


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def cal_one_mean_iou(image_array, label_array, num_parsing):
    hist = fast_hist(label_array, image_array, num_parsing).astype(np.float)
    num_cor_pix = np.diag(hist)
    num_gt_pix = hist.sum(1)
    union = num_gt_pix + hist.sum(0) - num_cor_pix
    iu = num_cor_pix / (num_gt_pix + hist.sum(0) - num_cor_pix)
    return iu


def parsing_on_boxes(parsing, rois, heatmap_size):
    device = rois.device
    rois = rois.to(torch.device("cpu"))
    parsing_list = []
    for i in range(rois.shape[0]):
        parsing_ins = parsing[i].cpu().numpy()
        xmin, ymin, xmax, ymax = torch.round(rois[i]).int()
        cropped_parsing = parsing_ins[max(0, ymin):ymax, max(0, xmin):xmax]
        resized_parsing = cv2.resize(
            cropped_parsing, (heatmap_size[1], heatmap_size[0]), interpolation=cv2.INTER_NEAREST
        )
        parsing_list.append(torch.from_numpy(resized_parsing))

    if len(parsing_list) == 0:
        return torch.empty(0, dtype=torch.int64, device=device)
    return torch.stack(parsing_list, dim=0).to(device, dtype=torch.int64)


def project_parsing_on_boxes(parsing, proposals, resolution):
    proposals = proposals.convert("xyxy")
    assert parsing.size == proposals.size, "{}, {}".format(parsing, proposals)

    return parsing_on_boxes(parsing.parsing, proposals.bbox, resolution)


class ParsingHeadLossComputation(object):
    def __init__(self, proposal_matcher, resolution, parsingiou_on):
        """
        Arguments:
            proposal_matcher (Matcher)
            resolution (tuple)
        """
        self.proposal_matcher = proposal_matcher
        self.resolution = resolution

        self.across_sample = cfg.AIParsing.ACROSS_SAMPLE
        self.roi_size_per_img = cfg.AIParsing.ROI_SIZE_PER_IMG
        self.parsingiou_on = parsingiou_on

    def match_targets_to_proposals(self, proposal, target):
        match_quality_matrix = boxlist_iou(target, proposal)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        target = target.copy_with_fields(["labels", "parsing"])

        matched_targets = target[matched_idxs.clamp(min=0)]
        matched_targets.add_field("matched_idxs", matched_idxs)
        return matched_targets

    def prepare_targets(self, proposals, targets):
        all_positive_proposals = []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            matched_targets = self.match_targets_to_proposals(
                proposals_per_image, targets_per_image
            )
            matched_idxs = matched_targets.get_field("matched_idxs")

            labels_per_image = matched_targets.get_field("labels")
            labels_per_image = labels_per_image.to(dtype=torch.int64)

            neg_inds = matched_idxs == Matcher.BELOW_LOW_THRESHOLD
            labels_per_image[neg_inds] = 0
            # parsing are only computed on positive samples
            positive_inds = torch.nonzero(labels_per_image > 0).squeeze(1)

            positive_proposals = proposals_per_image[positive_inds]
            _parsing = matched_targets.get_field("parsing")[positive_inds]

            parsing_per_image = project_parsing_on_boxes(
                _parsing, positive_proposals, self.resolution
            )
            positive_proposals.add_field("parsing_targets", parsing_per_image)
            all_positive_proposals.append(positive_proposals)

        return all_positive_proposals

    def resample(self, proposals, targets):
        # get all positive proposals (for single image on per GPU)
        positive_proposals = keep_only_positive_boxes(proposals)
        # resample for getting targets or matching new IoU
        positive_proposals = self.prepare_targets(positive_proposals, targets)
        # apply across-sample strategy (for a batch of images on per GPU)
        positive_proposals = across_sample(
            positive_proposals, roi_size_per_img=self.roi_size_per_img, across_sample=self.across_sample
        )

        self.positive_proposals = positive_proposals

        all_num_positive_proposals = 0
        for positive_proposals_per_image in positive_proposals:
            all_num_positive_proposals += len(positive_proposals_per_image)
        if all_num_positive_proposals == 0:
            positive_proposals = [proposals[0][:1]]
        return positive_proposals

    def __call__(self, parsing_logits, edge_logits):
        parsing_targets = [proposals_per_img.get_field("parsing_targets") for proposals_per_img in self.positive_proposals]
        parsing_targets = cat(parsing_targets, dim=0)

        #parsing loss
        if parsing_targets.numel() == 0:
            if not self.parsingiou_on:
                return parsing_logits.sum() * 0
            else:
                return parsing_logits.sum() * 0, None

        parsing_loss = F.cross_entropy(
            parsing_logits, parsing_targets, reduction="mean", ignore_index=255
        )

        parsing_logits_iou = F.softmax(parsing_logits, dim=1)
        parsing_iou_loss = L.lovasz_softmax(parsing_logits_iou, parsing_targets, ignore=255)
        
        # edge loss
        edges_labels = generate_edge_tensor(parsing_targets)

        pos_num = torch.sum(edges_labels == 1, dtype=torch.float)
        neg_num = torch.sum(edges_labels == 0, dtype=torch.float)

        weight_pos = neg_num / (pos_num + neg_num)
        weight_neg = pos_num / (pos_num + neg_num)
        weights = torch.tensor([weight_neg, weight_pos])  # edge loss weight
        edge_loss = F.cross_entropy(edge_logits, edges_labels.type(torch.cuda.LongTensor),weights.cuda(), ignore_index=255)

        #pdb.set_trace()
        # foreground loss
        # fg_labels = parsing_targets.clone()
        # fg_labels[(fg_labels>0) & (fg_labels<20)] = 1
        # fg_loss = F.cross_entropy(fg_logits, fg_labels, reduction="mean", ignore_index=255)

        # total_loss = parsing_loss + edge_loss + fg_loss
        # total_loss = parsing_loss + edge_loss
        # total_loss *= cfg.AIParsing.LOSS_WEIGHT
        # return total_loss

        if self.parsingiou_on:
            # TODO: use tensor for speeding up
            pred_parsings_np = parsing_logits.detach().argmax(dim=1).cpu().numpy()
            parsing_targets_np = parsing_targets.cpu().numpy()

            N = parsing_targets_np.shape[0]
            parsingiou_targets = np.zeros(N, dtype=np.float)

            for _ in range(N):
                parsing_iou = cal_one_mean_iou(parsing_targets_np[_], pred_parsings_np[_], cfg.AIParsing.NUM_PARSING)
                parsingiou_targets[_] = np.nanmean(parsing_iou)
            parsingiou_targets = torch.from_numpy(parsingiou_targets).to(parsing_targets.device, dtype=torch.float)


        parsing_loss *= cfg.AIParsing.LOSS_WEIGHT
        edge_loss *= cfg.AIParsing.LOSS_WEIGHT
        parsing_iou_loss *= cfg.AIParsing.LOSS_WEIGHT

        if not self.parsingiou_on:
            return parsing_loss, parsing_iou_loss, edge_loss
        else:
            return parsing_loss, parsing_iou_loss, edge_loss, parsingiou_targets
        


def parsing_loss_evaluator():
    matcher = Matcher(
        cfg.AIParsing.FG_IOU_THRESHOLD,
        cfg.AIParsing.BG_IOU_THRESHOLD,
        allow_low_quality_matches=False,
    )

    loss_evaluator = ParsingHeadLossComputation(
        matcher, cfg.AIParsing.RESOLUTION, cfg.AIParsing.PARSINGIOU_ON
    )
    return loss_evaluator
