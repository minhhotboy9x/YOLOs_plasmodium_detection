import os
from ultralytics import YOLO, FastSAM, SAM
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.metrics import ConfusionMatrix
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from customs.custom_val3 import CustomedDetectionValidator

from copy import deepcopy
import torch

if __name__ == "__main__":
    args = {
        "model": None,
        "mode": "val",
        # "data": "'datasets/pa3_malaria_7_classes_8_folds_temp1/folds/fold_0/data_0.yaml",
        "split": 'val',
        "batch": 2, 
        # "conf": 0.25,
        "rect": True,
        "iou": 0.5,
        "agnostic_nms": True,
        "project": "run_PA3_8_folds_temp4",
    }

    ori_names = ["Ring", "Trophozoite", "S", "G", "Healthy", "others", "difficult"]

    mapping_gt_pred = {
                    0: 0, # TJ
                    1: 1, # TA
                    2: 2, # S1
                    3: 2, # S2
                    4: 3, # G1
                    5: 3, # G25
                    6: 4, # Healthy
                    7: 5, # others
                    8: 6 # difficult
                    }
    
    ignore_gt_class = [6] # class after mapping_gt_pred

    ignore_FP_pair = {4:5, 
                      5:4} # class after mapping_gt_pred
    
    

    # model = YOLO('trained model/our data 8 folds temp1/11s_coco_fold0_PA3_ourdata_temp1_500_epochs.pt')
    # model.predict('datasets/final_malaria_full_class/train/images/444.jpg',
    #               save=True, show_labels=False)

    Parent_dir = 'datasets/malaria_9_classes_8_folds_temp4'
    models_dir = 'trained model/our data 8 folds temp3'
    for i, path in enumerate(sorted(os.listdir(models_dir))):
        data_yaml = os.path.join(Parent_dir, f'folds/fold_{i}/data_{i}.yaml')
        data_dict = check_det_dataset(data_yaml)

        model = YOLO(os.path.join(models_dir, path))
        model.model.names = data_dict['names']
        args['model'] = model.model
        args['data'] = data_yaml
        validator = CustomedDetectionValidator(
                                            ignore_gt_class=ignore_gt_class, 
                                            ignore_FP_pair=ignore_FP_pair,
                                            mapping_gt_pred=mapping_gt_pred,
                                            ori_names=ori_names,
                                            args = args)
        validator()
        # break
    
    # directory = "run_PA3_8_folds_temp1/test/false_detection"

    # for root, dirs, files in os.walk(directory):
    #     for file in files:
    #         if file.endswith("_All.jpg"):
    #             file_path = os.path.join(root, file)
    #             try:
    #                 os.remove(file_path)
    #                 print(f"Đã xóa: {file_path}")
    #             except Exception as e:
    #                 print(f"Lỗi khi xóa {file_path}: {e}")

