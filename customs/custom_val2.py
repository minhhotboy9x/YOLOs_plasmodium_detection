import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from pathlib import Path

import numpy as np
import torch
import cv2
import warnings
import matplotlib.pyplot as plt

from .custom_res import CustomedResults
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils.metrics import ConfusionMatrix, box_iou, batch_probiou
from ultralytics.models.yolo.detect import DetectionValidator


#---------------------------------------------------------- CustomedConfusionMatrix ----------------------------------------------------------#
class CustomedConfusionMatrix(ConfusionMatrix):
    """
    A class for calculating and updating a confusion matrix for object detection and classification tasks.

    Attributes:
        task (str): The type of task, either 'detect' or 'classify'.
        matrix (np.ndarray): The confusion matrix, with dimensions depending on the task.
        nc (int): The number of classes.
        conf (float): The confidence threshold for detections.
        iou_thres (float): The Intersection over Union threshold.
    """
    def __init__(self, ignore_gt_class:list, ignore_FP_pair:dict, names, save_dir, nc, conf=0.25, iou_thres=0.45, task="detect"):
        super().__init__(nc, conf, iou_thres, task)
        self.ignore_gt_class = ignore_gt_class
        self.ignore_FP_pair = ignore_FP_pair
        self.names = names
        self.save_dir = save_dir
        os.mkdir(self.save_dir / "false_detection")
    
    def process_batch(self, pbatch, detections, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
        device = gt_cls.device
        misdetected_detection = [] # result for false positive
        FP_labels = [] # result True label for false positive
        mismatched_gt = [] # result for false negative
        false_det_dir = self.save_dir / "false_detection" / f'{Path(pbatch["im_file"]).stem}_FP.jpg'
        false_gt_dir = self.save_dir / "false_detection" / f'{Path(pbatch["im_file"]).stem}_FN.jpg'
        all_det_dir = self.save_dir / "false_detection"  / f'{Path(pbatch["im_file"]).stem}_All.jpg'

        false_det_txt = self.save_dir / "txt_detection" / f'{Path(pbatch["im_file"]).stem}_FP.txt'
        all_det_txt = self.save_dir / "txt_detection" / f'{Path(pbatch["im_file"]).stem}_All.txt'

        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]
                detection_classes = detections[:, 5].int()
                for index, dc in enumerate(detection_classes):
                    self.matrix[dc, self.nc] += 1  # false positives

                    if self.nc in self.ignore_gt_class:
                        continue
                    if dc.item() in self.ignore_FP_pair.keys() \
                        and self.ignore_FP_pair[dc.item()] == self.nc:
                        continue

                    fp_back_ground_detection = detections[index].clone()
                    fp_back_ground_detection[4] = -fp_back_ground_detection[4]
                    misdetected_detection.append(fp_back_ground_detection)

            # plot false positive
            misdetected_detection = torch.stack(misdetected_detection) if misdetected_detection else None
            if misdetected_detection is not None :
                detect_res = CustomedResults(pbatch['ori_img'], pbatch['im_file'], self.names, 
                                    boxes=misdetected_detection, 
                                    gt_labels=['background'] * len(misdetected_detection))
                detect_res.save(false_det_dir, conf=False, line_width=3)
                detect_res.save_pred_gt_txt(false_det_txt, save_conf=True)

            # plot empty gt
            # gt_res = CustomedResults(pbatch['ori_img'], pbatch['im_file'], self.names)
            # gt_res.save(false_gt_dir, conf=True, line_width=4)

            # plot all detections
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]

            if detections:
                all_detect_res = Results(pbatch['ori_img'], pbatch['im_file'], self.names, boxes=detections)
                all_detect_res.save(all_det_dir, conf=True, line_width=3)
                all_detect_res.save_txt(all_det_txt)

            return
        
        if detections is None:
            gt_classes = gt_cls.int()
            for index, gc in enumerate(gt_classes):
                self.matrix[self.nc, gc] += 1  # background FN

                if gc.item() in self.ignore_gt_class:
                    continue
                if self.nc in self.ignore_FP_pair.keys() \
                    and self.ignore_FP_pair[self.nc] == gc.item():
                    continue

                mismatched_gt.append(np.hstack((gt_bboxes, torch.ones(1).to(device), gc)))

            # plot empty detection
            # detect_res = CustomedResults(pbatch['ori_img'], pbatch['im_file'], self.names)
            # detect_res.save(false_det_dir, conf=False, line_width=4)

            # plot gt
            mismatched_gt = torch.stack(mismatched_gt) if mismatched_gt else None
            gt_res = Results(pbatch['ori_img'], pbatch['im_file'], self.names, boxes=mismatched_gt)
            gt_res.save(false_gt_dir, conf=False, line_width=4)

            # plot all detections
            # all_detect_res = Results(pbatch['ori_img'], pbatch['im_file'], self.names, boxes=detections)
            # all_detect_res.save(all_det_dir, conf=True, line_width=3)
            return

        detections = detections[detections[:, 4] > self.conf]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        )

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # uniquify by detection
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # uniquify by gt
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i # a list true/false for each gt and detection
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
                if detection_classes[m1[j]] != gc:

                    if gc.item() in self.ignore_gt_class:
                        continue
                    if detection_classes[m1[j]].item() in self.ignore_FP_pair.keys() \
                        and self.ignore_FP_pair[detection_classes[m1[j]].item()] == gc.item():
                        continue

                    misdetected_detection.append(detections[m1[j]].squeeze(0))
                    mismatched_gt.append(torch.hstack((gt_bboxes[i], torch.ones(1).to(device), gc)))
                    FP_labels.append(self.names[int(gc.item())])
            else:
                self.matrix[self.nc, gc] += 1  # true background
                mismatched_gt.append(torch.hstack((gt_bboxes[i], torch.ones(1).to(device), gc)))

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # predicted background

                    if self.nc in self.ignore_gt_class:
                        continue
                    if dc.item() in self.ignore_FP_pair.keys() \
                        and self.ignore_FP_pair[dc.item()] == self.nc:
                        continue

                    fp_back_ground_detection = detections[i].clone()
                    fp_back_ground_detection[4] = -fp_back_ground_detection[4]
                    misdetected_detection.append(fp_back_ground_detection)
                    FP_labels.append('background')

        mismatched_gt = torch.stack(mismatched_gt) if mismatched_gt else None
        misdetected_detection = torch.stack(misdetected_detection) if misdetected_detection else None

        if misdetected_detection is not None:
            detect_res = CustomedResults(pbatch['ori_img'], pbatch['im_file'], 
                                        self.names, 
                                        boxes=misdetected_detection,
                                        gt_labels=FP_labels)
            detect_res.save(false_det_dir, conf=False, line_width=3)
            detect_res.save_pred_gt_txt(false_det_txt, )

        gt_res = Results(pbatch['ori_img'], pbatch['im_file'], self.names, boxes=mismatched_gt)
        gt_res.save(false_gt_dir, conf=True, line_width=4)

        all_detect_res = Results(pbatch['ori_img'], pbatch['im_file'], self.names, boxes=detections)
        all_detect_res.save(all_det_dir, conf=True, line_width=3)
        all_detect_res.save_txt(all_det_txt)
    
    @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", names=(), on_plot=None):
        """
        Plot the confusion matrix using seaborn and save it to a file.

        Args:
            normalize (bool): Whether to normalize the confusion matrix.
            save_dir (str): Directory where the plot will be saved.
            names (tuple): Names of classes, used as labels on the plot.
            on_plot (func): An optional callback to pass plots path and data when they are rendered.
        """
        import seaborn  # scope for faster 'import ultralytics'

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)  # normalize columns
        array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

        fig, ax = plt.subplots(1, 1, figsize=(12, 9), tight_layout=True)
        nc, nn = self.nc, len(names)  # number of classes, names
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < nn < 99) and (nn == nc)  # apply names to ticklabels
        ticklabels = (list(names) + ["background"]) if labels else "auto"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
            seaborn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 16},
                cmap="Blues",
                fmt=".2f" if normalize else ".0f",
                square=True,
                vmin=0.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))
        title = "Confusion Matrix" + " Normalized" * normalize
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(title)
        plot_fname = Path(save_dir) / f'{title.lower().replace(" ", "_")}.png'
        fig.savefig(plot_fname, dpi=250)
        plt.close(fig)
        if on_plot:
            on_plot(plot_fname)

    

#---------------------------------------------------------- CustomedDetectionValidator ----------------------------------------------------------#    
class CustomedDetectionValidator(DetectionValidator):
    """
    A class extending the BaseValidator class for validation based on a detection model.

    Example:
        ```python
        from ultralytics.models.yolo.detect import DetectionValidator

        args = dict(model="yolov8n.pt", data="coco8.yaml")
        validator = DetectionValidator(args=args)
        validator()
        ```
    """
    def __init__(self, ignore_gt_class:list = {}, ignore_FP_pair:dict = {}, dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.ignore_gt_class = ignore_gt_class
        self.ignore_FP_pair = ignore_FP_pair

    def init_metrics(self, model):
        super().init_metrics(model)
        self.confusion_matrix = CustomedConfusionMatrix(
            ignore_gt_class=self.ignore_gt_class,
            ignore_FP_pair=self.ignore_FP_pair,
            names=self.names, 
            save_dir=self.save_dir, 
            nc=self.nc, conf=self.args.conf)
    
    def _prepare_batch(self, si, batch):
        """Prepares a batch of images and annotations for validation."""
        idx = batch["batch_idx"] == si
        cls = batch["cls"][idx].squeeze(-1)
        bbox = batch["bboxes"][idx]
        ori_shape = batch["ori_shape"][si]
        imgsz = batch["img"].shape[2:]
        ratio_pad = batch["ratio_pad"][si]
        im_file = batch["im_file"][si]
        ori_img = cv2.imread(im_file)
        if len(cls):
            bbox = ops.xywh2xyxy(bbox) * torch.tensor(imgsz, device=self.device)[[1, 0, 1, 0]]  # target boxes
            ops.scale_boxes(imgsz, bbox, ori_shape, ratio_pad=ratio_pad)  # native-space labels
        return {"cls": cls, "bbox": bbox, "ori_shape": ori_shape, 
                "imgsz": imgsz, "ratio_pad": ratio_pad,
                "im_file": im_file,
                "ori_img": ori_img,}

    def filter_boxes_near_border(self, boxes, img_shape, margin=10):
        """
        Loại bỏ các bounding box nằm gần biên ảnh.
        
        Args:
            boxes (torch.Tensor): Tensor dạng (N, 6) với định dạng (x1, y1, x2, y2, conf, class)
            img_shape (tuple): Kích thước ảnh (H, W)
            margin (int): Khoảng cách tối thiểu từ biên để box được giữ lại
        
        Returns:
            torch.Tensor: Các bounding boxes hợp lệ
        """
        H, W = img_shape  # Chiều cao, chiều rộng ảnh

        # Điều kiện: box phải nằm cách biên ít nhất 'margin' pixels
        valid_mask = (boxes[:, 0] >= margin) & (boxes[:, 1] >= margin) & \
                    (boxes[:, 2] <= W - margin) & (boxes[:, 3] <= H - margin)

        return boxes[valid_mask]  # Chỉ giữ các box hợp lệ

    def update_metrics(self, preds, batch):
        """Metrics."""
        for si, pred in enumerate(preds):
            self.seen += 1
            npr = len(pred)
            stat = dict(
                conf=torch.zeros(0, device=self.device),
                pred_cls=torch.zeros(0, device=self.device),
                tp=torch.zeros(npr, self.niou, dtype=torch.bool, device=self.device),
            )
            pbatch = self._prepare_batch(si, batch)
            cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
            nl = len(cls)
            stat["target_cls"] = cls
            stat["target_img"] = cls.unique()
            if npr == 0:
                if nl:
                    for k in self.stats.keys():
                        self.stats[k].append(stat[k])
                    if self.args.plots:
                        # self.confusion_matrix.process_batch(detections=None, gt_bboxes=bbox, gt_cls=cls)
                        self.confusion_matrix.process_batch(pbatch, detections=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            predn = self.filter_boxes_near_border(predn, pbatch['ori_shape'])
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    # self.confusion_matrix.process_batch(predn, bbox, cls)
                    self.confusion_matrix.process_batch(pbatch, predn, bbox, cls)
            for k in self.stats.keys():
                self.stats[k].append(stat[k])

            # Save
            if self.args.save_json:
                self.pred_to_json(predn, batch["im_file"][si])
            if self.args.save_txt:
                self.save_one_txt(
                    predn,
                    self.args.save_conf,
                    pbatch["ori_shape"],
                    self.save_dir / "labels" / f'{Path(batch["im_file"][si]).stem}.txt',
                )

if __name__ == "__main__":
    model = YOLO('trained model/our data/v11s_coco_malaria_PA3.1_7_classes_500epochs.pt')

    # model.val(data='coco8.yaml', batch=2, plots=True, save_conf=True, save_crop=True, save_txt=True, conf=0.25)
    
    # data = check_det_dataset('datasets/pa3.1_malaria_7_classes/data.yaml')
    # model.model.nc = data['nc']
    # model.model.names = data['names']
    print(model.model.names)
    args = {
        "model": model.model,
        "mode": "val",
        "data": "datasets/pa3_malaria_7_classes/data.yaml",
        "split": 'test',
        "batch": 2, 
        # "conf": 0.25,
        "rect": True,
        "agnostic_nms": True
    }
    
    validator = CustomedDetectionValidator(args=args)
    # validator = DetectionValidator(args=args)
    validator()

