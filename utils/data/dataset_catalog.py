import os.path as osp

# Root directory of project
ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Path to data dir
_DATA_DIR = osp.abspath(osp.join(ROOT_DIR, 'datasets'))

# Required dataset entry keys
_IM_DIR = 'image_directory'
_ANN_FN = 'annotation_file'

# Available datasets
COMMON_DATASETS = {
    'coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2017.json',
    },
    'coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
    },
    'coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
    },
    'keypoints_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'keypoints_coco_2017_test-dev': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
    },
    'dense_coco_2017_train': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_train2017.json',
    },
    'dense_coco_2017_val': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_val2017.json',
    },
    'dense_coco_2017_test': {
        _IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        _ANN_FN:
            _DATA_DIR + '/coco/annotations/DensePoseData/densepose_coco_test.json',
    },
    'CIHP_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/train_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_train.json',
    },
    'CIHP_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/val_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_val.json',
    },
    'CIHP_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/CIHP/test_img',
        _ANN_FN:
            _DATA_DIR + '/CIHP/annotations/CIHP_test.json',
    },
    'LV-MHP-v2_train': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/LV-MHP-v2/train/train_img',
        _ANN_FN:
            _DATA_DIR + '/LV-MHP-v2/annotations/LV_MHP_V2_train.json',
    },
    'LV-MHP-v2_val': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/LV-MHP-v2/val/val_img',
        _ANN_FN:
            _DATA_DIR + '/LV-MHP-v2/annotations/LV_MHP_V2_val.json',
    },
    'LV-MHP-v2_test': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/LV-MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/LV-MHP-v2/annotations/MHP-v2_test_all.json',
    },
    'LV-MHP-v2_test_inter_top10': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/LV-MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/LV-MHP-v2/annotations/MHP-v2_test_inter_top10.json',
    },
    'LV-MHP-v2_test_inter_top20': {  # new addition by wzh
        _IM_DIR:
            _DATA_DIR + '/LV-MHP-v2/test_img',
        _ANN_FN:
            _DATA_DIR + '/LV-MHP-v2/annotations/MHP-v2_test_inter_top20.json',
    },
    'PASCAL-Person-Part_train': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/train_img',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/pascal_person_part_train.json',
    },
    'PASCAL-Person-Part_test': {  # new addition by soeaver
        _IM_DIR:
            _DATA_DIR + '/PASCAL-Person-Part/test_img',
        _ANN_FN:
            _DATA_DIR + '/PASCAL-Person-Part/annotations/pascal_person_part_test.json',
    }
}
