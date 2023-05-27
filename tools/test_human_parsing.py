# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import time
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
from maskrcnn_benchmark.data.evaluation import post_processing
from maskrcnn_benchmark.data.evaluation import evaluation
import utils.vis as vis_utils
from maskrcnn_benchmark.utils.imports import import_file
from maskrcnn_benchmark.data.transforms import build_transforms
from maskrcnn_benchmark.data.build import build_dataset

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
    parser.add_argument("--input-file", type=str, default='datasets/val_id.txt')
    parser.add_argument("--input-folder", type=str, default='/home/zhangsy/zsy/parsercnn/Parsing-R-CNN/data/CIHP/val_img/')
    parser.add_argument("--output-folder", type=str, default='./parsing_out')
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

    #print(args.output_folder)
    cfg.merge_from_file(args.config_file)
    #cfg.merge_from_list(args.opts)
    cfg.freeze()

    paths_catalog = import_file(
        "maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True
    )
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TEST

    transforms = build_transforms(cfg, is_train=False)
    dataset = build_dataset(dataset_list, transforms, DatasetCatalog, is_train=False)
    dataset = dataset[0]   



    save_dir = ""
    logger = setup_logger("maskrcnn_benchmark", save_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)





    file_list = [i_id.strip() for i_id in open(args.input_file)]

    import pdb

    start = time.time()
    num_img = cfg.TEST.IMS_PER_BATCH
    start_ind = 0
    end_ind = len(file_list)
    with torch.no_grad():
        all_boxes = []
        all_segms = []
        all_keyps = []
        all_parss = []
        all_pscores = []
        all_uvs = []
        #i=0
    
    
        for i in range(start_ind, end_ind, num_img):
            ims1 = []
            image_ids = []
            for j in range(i, i + num_img):
                if j == end_ind:
                    break
                im1 = dataset.pull_image(j)
                ims1.append(im1)
                image_ids.append(j)

        #for file in file_list:
            ims = []
            filenames = []

            file = file_list[i]
            i = i + 1
            #print(os.path.join(args.input_folder, file+'.jpg'))
            im = cv2.imread(os.path.join(args.input_folder, file+'.jpg'), cv2.IMREAD_COLOR)
            ims.append(im)            
            filenames.append(file+'.jpg')

            
            result, features = parsing_test.im_detect_bbox(model, ims1)

            result = parsing_test.im_detect_parsing(model, result, features)


            eval_results, ims_results = post_processing(result, ims1,filenames,args.output_folder,dataset,image_ids)
            box_results, seg_results, kpt_results, par_results, par_score, uvs_results = eval_results
            ims_dets, ims_labels, ims_segs, ims_kpts, ims_pars, ims_uvs = ims_results
            #pdb.set_trace()
            if(len(box_results)==0):
                print(file+'.jpg')
                print('boxes:',len(box_results))
            all_boxes += box_results
            all_segms += seg_results
            all_keyps += kpt_results
            all_parss += par_results
            all_pscores += par_score
            all_uvs += uvs_results
            result=None
            features=None


    end = time.time()

    print(end-start)
    evaluation(dataset, all_boxes, all_segms, all_keyps, all_parss, all_pscores, all_uvs)
if __name__ == "__main__":
    main()
