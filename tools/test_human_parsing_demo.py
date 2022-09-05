# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.parsing_inference import run_inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
import maskrcnn_benchmark.engine.test_demo as parsing_test
import cv2
import pdb
from maskrcnn_benchmark.data.evaluation_demo import post_processing_demo
import utils.vis as vis_utils

def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument('--range', help='start (inclusive) and end (exclusive) indices', type=int, nargs=2)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--input_file", type=str, default='1.txt')
    parser.add_argument("--input_folder", type=str, default='./input_jpgs/')
    parser.add_argument("--output_folder", type=str, default='./output_jpgs/')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    multi_gpu_testing = True if num_gpus > 1 else False

    distributed = num_gpus > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    print(args.output_folder)
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)



    #file_list = [i_id.strip() for i_id in open(args.input_file)]


    import pdb
    print(args.input_folder)
    file_list = os.listdir(args.input_folder)



    with torch.no_grad():

        for file in file_list:
            ims = []
            filenames = []
            im = cv2.imread(os.path.join(args.input_folder, file.strip()), cv2.IMREAD_COLOR)
            ims.append(im)            
            filenames.append(file)
            print(file)
            result, features = parsing_test.im_detect_bbox(model, ims)

            result = parsing_test.im_detect_parsing(model, result, features)

            ims_dets, ims_labels, ims_pars = post_processing_demo(result, ims,filenames,args.output_folder)
            # box_results, seg_results, kpt_results, par_results, par_score, uvs_results = eval_results
            # ims_dets, ims_labels, ims_segs, ims_kpts, ims_pars, ims_uvs = ims_results
            parsing_box_ins_dir = os.path.join(args.output_folder, 'parsing_box_ins_vis')
            if not os.path.exists(parsing_box_ins_dir):
                os.makedirs(parsing_box_ins_dir)

            if cfg.VIS.ENABLED:
                for k, im in enumerate(ims):
                    if len(ims_dets) == 0:
                        break
                    im_name = file
                    vis_im = vis_utils.vis_one_image_opencv(
                        im,
                        cfg,
                        ims_dets[k],
                        ims_labels[k],
                        segms=None,
                        keypoints=None,
                        parsing=ims_pars[k],
                        uv=None,
                        dataset=None,
                    )
                    
                    cv2.imwrite(os.path.join(parsing_box_ins_dir, '{}'.format(file)), vis_im)


if __name__ == "__main__":
    main()
