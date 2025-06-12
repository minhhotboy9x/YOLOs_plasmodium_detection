import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.append(parent_dir)

from pathlib import Path

import json
import numpy as np
import torch
import torchvision
import cv2
import warnings
import matplotlib.pyplot as plt

from .custom_res import CustomedResults
from .utils import *
from ultralytics import YOLO
from ultralytics.nn.autobackend import AutoBackend
from ultralytics.engine.results import Results
from ultralytics.utils import ops
from ultralytics.utils import LOGGER, SimpleClass, TryExcept, plt_settings, TQDM, callbacks, colorstr, emojis
from ultralytics.utils.metrics import ConfusionMatrix, box_iou, batch_probiou, DetMetrics
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils.ops import xywh2xyxy, nms_rotated, xyxy2xywh, scale_boxes
from ultralytics.utils.checks import check_imgsz
from ultralytics.data.utils import check_cls_dataset, check_det_dataset
from ultralytics.utils.ops import Profile
from ultralytics.utils.torch_utils import de_parallel, select_device, smart_inference_mode

def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nc=0,  # number of classes (optional)
    max_time_img=0.05,
    max_nms=30000,
    max_wh=7680,
    in_place=True,
    rotated=False,
    end2end=False,
):
    """
    Perform non-maximum suppression (NMS) on a set of boxes, with support for masks and multiple labels per box.

    Args:
        prediction (torch.Tensor): A tensor of shape (batch_size, num_classes + 4 + num_masks, num_boxes)
            containing the predicted boxes, classes, and masks. The tensor should be in the format
            output by a model, such as YOLO.
        conf_thres (float): The confidence threshold below which boxes will be filtered out.
            Valid values are between 0.0 and 1.0.
        iou_thres (float): The IoU threshold below which boxes will be filtered out during NMS.
            Valid values are between 0.0 and 1.0.
        classes (List[int]): A list of class indices to consider. If None, all classes will be considered.
        agnostic (bool): If True, the model is agnostic to the number of classes, and all
            classes will be considered as one.
        multi_label (bool): If True, each box may have multiple labels.
        labels (List[List[Union[int, float, torch.Tensor]]]): A list of lists, where each inner
            list contains the apriori labels for a given image. The list should be in the format
            output by a dataloader, with each label being a tuple of (class_index, x1, y1, x2, y2).
        max_det (int): The maximum number of boxes to keep after NMS.
        nc (int): The number of classes output by the model. Any indices after this will be considered masks.
        max_time_img (float): The maximum time (seconds) for processing one image.
        max_nms (int): The maximum number of boxes into torchvision.ops.nms().
        max_wh (int): The maximum box width and height in pixels.
        in_place (bool): If True, the input prediction tensor will be modified in place.
        rotated (bool): If Oriented Bounding Boxes (OBB) are being passed for NMS.
        end2end (bool): If the model doesn't require NMS.

    Returns:
        (List[torch.Tensor]): A list of length batch_size, where each element is a tensor of
            shape (num_boxes, 6 + num_masks) containing the kept boxes, with columns
            (x1, y1, x2, y2, confidence, class, mask1, mask2, ...).
    """

    # Checks
    assert 0 <= conf_thres <= 1, f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert 0 <= iou_thres <= 1, f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"
    if isinstance(prediction, (list, tuple)):  # YOLOv8 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output
    if classes is not None:
        classes = torch.tensor(classes, device=prediction.device)

    if prediction.shape[-1] == 6 or end2end:  # end-to-end model (BNC, i.e. 1,300,6)
        output = [pred[pred[:, 4] > conf_thres][:max_det] for pred in prediction]
        if classes is not None:
            output = [pred[(pred[:, 5:6] == classes).any(1)] for pred in output]
        return output

    bs = prediction.shape[0]  # batch size (BCN, i.e. 1,84,6300)
    nc = nc or (prediction.shape[1] - 4)  # number of classes
    nm = prediction.shape[1] - nc - 4  # number of masks
    mi = 4 + nc  # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thres  # candidates

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    time_limit = 2.0 + max_time_img * bs  # seconds to quit after
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    prediction = prediction.transpose(-1, -2)  # shape(1,84,6300) to shape(1,6300,84)
    if not rotated:
        if in_place:
            prediction[..., :4] = xywh2xyxy(prediction[..., :4])  # xywh to xyxy
        else:
            prediction = torch.cat((xywh2xyxy(prediction[..., :4]), prediction[..., 4:]), dim=-1)  # xywh to xyxy

    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs # [tensor([], size=(0, 6))] for box, class, score
    output1 = [torch.zeros((0, 4 + nc), device=prediction.device)] * bs # [tensor([], size=(0, 4 + nc))] for box, score1 ,score2, ..., scoreN
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[:, 2:4] < min_wh) | (x[:, 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]) and not rotated:
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 4), device=x.device)
            v[:, :4] = xywh2xyxy(lb[:, 1:5])  # box
            v[range(len(lb)), lb[:, 0].long() + 4] = 1.0  # cls
            x = torch.cat((x, v), 0)
        x1 = x.clone()
        # If none remain process next image
        if not x.shape[0]:
            continue

        # Detections matrix nx6 (xyxy, conf, cls)
        box, cls, mask = x.split((4, nc, nm), 1)

        if multi_label:
            i, j = torch.where(cls > conf_thres)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float(), mask[i]), 1) # [N, box, score, class, mask]
            x1 = torch.cat((box[i],               # [N, 4]
                            cls[i],               # [N, nc]
                            ), 1) # [N, box, score1, score2, ..., scoreN]
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            conf_mask = conf.view(-1) > conf_thres
            x = torch.cat((box, conf, j.float(), mask), 1)[conf_mask] # [N, box, score, class, mask]
            x1 = torch.cat((box, cls), 1)[conf_mask]  # [box, score1, score2, ..., scoreN]

        # Filter by class
        if classes is not None:
            class_mask = (x[:, 5:6] == classes).any(1)
            x = x[class_mask]
            x1 = x1[class_mask]  # align x1 theo class_mask

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        if n > max_nms:  # excess boxes
            topk = x[:, 4].argsort(descending=True)[:max_nms] # sort by confidence and remove excess boxes
            x = x[topk]
            x1 = x1[topk]  # align x1 theo top confidence  

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        scores = x[:, 4]  # scores
        if rotated:
            boxes = torch.cat((x[:, :2] + c, x[:, 2:4], x[:, -1:]), dim=-1)  # xywhr
            i = nms_rotated(boxes, scores, iou_thres)
        else:
            boxes = x[:, :4] + c  # boxes (offset by class)
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        i = i[:max_det]  # limit detections

        output[xi] = x[i]
        output1[xi] = x1[i]  # align output1 theo i
    return output, output1

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
    def __init__(self, names, save_dir, 
                 nc, model_nc, conf=0.25, iou_thres=0.45, task="detect",
                 ignore_gt_class:list = [], ignore_FP_pair:dict = {},
                 mapping_gt_pred:dict = {}, ori_names:list = []):
        super().__init__(nc, conf, iou_thres, task)
        self.ignore_gt_class = ignore_gt_class
        self.ignore_FP_pair = ignore_FP_pair
        self.mapping_gt_pred = mapping_gt_pred
        self.ori_names = ori_names
        self.names = names
        self.model_nc = model_nc
        self.save_dir = save_dir
        self.matrix = np.zeros((model_nc + 1, nc + 1)) if self.task == "detect" else np.zeros((model_nc, nc))
        os.mkdir(self.save_dir / "false_detection")
    
    def process_batch(self, pbatch, detections, detections1, gt_bboxes, gt_cls):
        """
        Update confusion matrix for object detection task.

        Args:
            detections (Array[N, 6] | Array[N, 7]): Detected bounding boxes and their associated information.
                                      Each row should contain (x1, y1, x2, y2, conf, class)
                                      or with an additional element `angle` when it's obb.
            detections1 (Array[N, 4 + C]): Detected bounding boxes and their associated class scores.
            gt_bboxes (Array[M, 4]| Array[N, 5]): Ground truth bounding boxes with xyxy/xyxyr format.
            gt_cls (Array[M]): The class labels.
        """
        device = gt_cls.device
        misdetected_detection = [] # result for false positive
        FP_labels = [] # result True label for false positive
        mismatched_gt = [] # result for false negative

        full_cls_detections = [] # result for full class detection
        matched_gt_names = []

        false_det_dir = self.save_dir / "false_detection" / f'{Path(pbatch["im_file"]).stem}_FP.jpg'
        false_gt_dir = self.save_dir / "false_detection" / f'{Path(pbatch["im_file"]).stem}_FN.jpg'
        all_det_dir = self.save_dir / "false_detection"  / f'{Path(pbatch["im_file"]).stem}_All.jpg'

        false_det_txt = self.save_dir / "txt_detection" / f'{Path(pbatch["im_file"]).stem}_FP.txt'
        all_det_txt = self.save_dir / "txt_detection" / f'{Path(pbatch["im_file"]).stem}_All.txt'

        det_gt_txt = self.save_dir / "det_gt" / f'{Path(pbatch["im_file"]).stem}_.txt'

        if gt_cls.shape[0] == 0:  # Check if labels is empty
            if detections is not None:
                masked_detections = detections[:, 4] > self.conf
                detections = detections[masked_detections]
                detections1 = detections1[masked_detections] 
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

                    full_cls_detections.append(detections1[index].clone())

            # plot false positive
            misdetected_detection = torch.stack(misdetected_detection) if misdetected_detection else None
            if misdetected_detection is not None :
                detect_res = CustomedResults(
                                    pbatch['ori_img'], pbatch['im_file'], 
                                    self.ori_names, 
                                    boxes=misdetected_detection, 
                                    gt_labels=['background'] * len(misdetected_detection))
                detect_res.save(false_det_dir, conf=False, line_width=3)
                detect_res.save_pred_gt_txt(false_det_txt, save_conf=True)

                save_full_det_gt(det_gt_txt, full_cls_detections, ['background'] * len(misdetected_detection), self.ori_names)

            # plot empty gt
            # gt_res = CustomedResults(pbatch['ori_img'], pbatch['im_file'], self.names)
            # gt_res.save(false_gt_dir, conf=True, line_width=4)

            # plot all detections
            if detections is not None:
                detections = detections[detections[:, 4] > self.conf]

            if detections:
                all_detect_res = Results(pbatch['ori_img'], pbatch['im_file'], self.ori_names, boxes=detections)
                all_detect_res.save(all_det_dir, conf=True, line_width=3)
                all_detect_res.save_txt(all_det_txt)

            return
        
        if detections is None:
            gt_classes = gt_cls.int()
            for index, gc in enumerate(gt_classes):
                self.matrix[self.model_nc, gc] += 1  # background FN

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

        masked_detections = detections[:, 4] > self.conf
        detections = detections[masked_detections]
        detections1 = detections1[masked_detections]
        gt_classes = gt_cls.int()
        detection_classes = detections[:, 5].int()
        is_obb = detections.shape[1] == 7 and gt_bboxes.shape[1] == 5  # with additional `angle` dimension
        iou = (
            batch_probiou(gt_bboxes, torch.cat([detections[:, :4], detections[:, -1:]], dim=-1))
            if is_obb
            else box_iou(gt_bboxes, detections[:, :4])
        ) #[M, N] where M is number of gt and N is number of detections

        x = torch.where(iou > self.iou_thres) # x[0] is the index of gt, x[1] is the index of detection
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy() 
            # [M, 3] where M is number of matches, gt index, detection index, iou score
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]] # uniquify by detection
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]] # uniquify by gt
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int) # m0 is gt index, m1 is detection index, _ is iou score
        for i, gc in enumerate(gt_classes):
            j = m0 == i # a list true/false for each gt and detection
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct

                mapped_gc = gc
                if gc.item() in self.mapping_gt_pred: # check if gt class is in mapping_gt_pred
                    mapped_gc = self.mapping_gt_pred[gc.item()]

                if detection_classes[m1[j]].item() != mapped_gc:

                    if mapped_gc in self.ignore_gt_class:
                        continue
                    if detection_classes[m1[j]].item() in self.ignore_FP_pair.keys() \
                        and self.ignore_FP_pair[detection_classes[m1[j]].item()] == mapped_gc:
                        continue

                    misdetected_detection.append(detections[m1[j]].squeeze(0))
                    mismatched_gt.append(torch.hstack((gt_bboxes[i], torch.ones(1).to(device), gc)))
                    FP_labels.append(self.names[int(gc.item())])
                
                # save matched detection and gt class
                full_cls_detections.append(detections1[m1[j]].squeeze(0).clone())
                matched_gt_names.append(self.names[int(gc.item())])
            else:
                self.matrix[self.model_nc, gc] += 1  # true background
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

                    # save matched detection and gt class
                    full_cls_detections.append(detections1[m1[j]].squeeze(0).clone())
                    matched_gt_names.append('background')

        mismatched_gt = torch.stack(mismatched_gt) if mismatched_gt else None
        misdetected_detection = torch.stack(misdetected_detection) if misdetected_detection else None

        if misdetected_detection is not None:
            detect_res = CustomedResults(pbatch['ori_img'], pbatch['im_file'], 
                                        self.ori_names, 
                                        boxes=misdetected_detection,
                                        gt_labels=FP_labels)
            detect_res.save(false_det_dir, conf=False, line_width=3)
            detect_res.save_pred_gt_txt(false_det_txt, )

        gt_res = Results(pbatch['ori_img'], pbatch['im_file'], self.names, boxes=mismatched_gt)
        gt_res.save(false_gt_dir, conf=True, line_width=4)

        all_detect_res = Results(pbatch['ori_img'], pbatch['im_file'], self.ori_names, boxes=detections)
        all_detect_res.save(all_det_dir, conf=True, line_width=3)
        all_detect_res.save_txt(all_det_txt)

        # save full class detection
        save_full_det_gt(det_gt_txt, full_cls_detections, matched_gt_names, self.ori_names)

    # @TryExcept("WARNING ⚠️ ConfusionMatrix plot failure")
    @plt_settings()
    def plot(self, normalize=True, save_dir="", model_names=(), label_names=(), on_plot=None):
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
        nc, model_nn, label_nn = self.nc, len(model_names), len(label_names)  # number of classes, names
        seaborn.set_theme(font_scale=1.0 if nc < 50 else 0.8)  # for label size
        labels = (0 < label_nn < 99) and (label_nn == nc)  # apply names to ticklabels
        xticklabels = (list(label_names) + ["background"]) if labels else "auto"
        yticklabels = (list(model_names) + ["background"]) if labels else "auto"
        
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
                xticklabels=xticklabels,
                yticklabels=yticklabels,
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
    def __init__(self, ignore_gt_class:list = {}, ignore_FP_pair:dict = {}, 
                 mapping_gt_pred:dict = {}, ori_names:list = [],
                 dataloader=None, save_dir=None, pbar=None, args=None, _callbacks=None):
        super().__init__(dataloader, save_dir, pbar, args, _callbacks)
        self.ignore_gt_class = ignore_gt_class
        self.ignore_FP_pair = ignore_FP_pair
        self.mapping_gt_pred = mapping_gt_pred
        self.ori_names = ori_names
        data_dict = check_det_dataset(self.args.data)
        self.num_gt_classes = data_dict['nc']
        self.gt_names = data_dict['names']
        self.metrics.names = self.gt_names
    
    @smart_inference_mode()
    def __call__(self, trainer=None, model=None):
        """
        Execute validation process, running inference on dataloader and computing performance metrics.

        Args:
            trainer (object, optional): Trainer object that contains the model to validate.
            model (nn.Module, optional): Model to validate if not using a trainer.

        Returns:
            stats (dict): Dictionary containing validation statistics.
        """
        self.training = trainer is not None
        augment = self.args.augment and (not self.training)
        if self.training:
            self.device = trainer.device
            self.data = trainer.data
            # Force FP16 val during training
            self.args.half = self.device.type != "cpu" and trainer.amp
            model = trainer.ema.ema or trainer.model
            model = model.half() if self.args.half else model.float()
            # self.model = model
            self.loss = torch.zeros_like(trainer.loss_items, device=trainer.device)
            self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
            model.eval()
        else:
            if str(self.args.model).endswith(".yaml") and model is None:
                LOGGER.warning("WARNING ⚠️ validating an untrained model YAML will result in 0 mAP.")
            callbacks.add_integration_callbacks(self)
            model = AutoBackend(
                weights=model or self.args.model,
                device=select_device(self.args.device, self.args.batch),
                dnn=self.args.dnn,
                data=self.args.data,
                fp16=self.args.half,
            )
            # self.model = model
            self.device = model.device  # update device
            self.args.half = model.fp16  # update half
            stride, pt, jit, engine = model.stride, model.pt, model.jit, model.engine
            imgsz = check_imgsz(self.args.imgsz, stride=stride)
            if engine:
                self.args.batch = model.batch_size
            elif not pt and not jit:
                self.args.batch = model.metadata.get("batch", 1)  # export.py models default to batch-size 1
                LOGGER.info(f"Setting batch={self.args.batch} input of shape ({self.args.batch}, 3, {imgsz}, {imgsz})")

            if str(self.args.data).split(".")[-1] in {"yaml", "yml"}:
                self.data = check_det_dataset(self.args.data)
            elif self.args.task == "classify":
                self.data = check_cls_dataset(self.args.data, split=self.args.split)
            else:
                raise FileNotFoundError(emojis(f"Dataset '{self.args.data}' for task={self.args.task} not found ❌"))

            if self.device.type in {"cpu", "mps"}:
                self.args.workers = 0  # faster CPU val as time dominated by inference, not dataloading
            if not pt:
                self.args.rect = False
            self.stride = model.stride  # used in get_dataloader() for padding
            self.dataloader = self.dataloader or self.get_dataloader(self.data.get(self.args.split), self.args.batch)

            model.eval()
            model.warmup(imgsz=(1 if pt else self.args.batch, 3, imgsz, imgsz))  # warmup

        self.run_callbacks("on_val_start")
        dt = (
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
            Profile(device=self.device),
        )
        bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
        self.init_metrics(de_parallel(model))
        self.jdict = []  # empty before each val
        for batch_i, batch in enumerate(bar):
            self.run_callbacks("on_val_batch_start")
            self.batch_i = batch_i
            # Preprocess
            with dt[0]:
                batch = self.preprocess(batch)

            # Inference
            with dt[1]:
                preds = model(batch["img"], augment=augment)

            # Loss
            with dt[2]:
                if self.training:
                    self.loss += model.loss(batch, preds)[1]

            # Postprocess
            with dt[3]:
                preds, preds1 = self.postprocess(preds)

            self.update_metrics(preds, preds1, batch)
            if self.args.plots and batch_i < 3:
                self.plot_val_samples(batch, batch_i)
                self.plot_predictions(batch, preds, batch_i)

            self.run_callbacks("on_val_batch_end")
        stats = self.get_stats()
        self.check_stats(stats)
        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
        self.finalize_metrics()
        self.print_results()
        self.run_callbacks("on_val_end")
        if self.training:
            model.float()
            results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
            return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
        else:
            LOGGER.info(
                "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
                    *tuple(self.speed.values())
                )
            )
            if self.args.save_json and self.jdict:
                with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
                    LOGGER.info(f"Saving {f.name}...")
                    json.dump(self.jdict, f)  # flatten and save
                stats = self.eval_json(stats)  # update stats
            if self.args.plots or self.args.save_json:
                LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
            return stats

    def init_metrics(self, model):
        super().init_metrics(model)
        self.nc = model.model.yaml["nc"]  # nu
        self.init_names = self.names # keep model names
        self.names = self.gt_names
        self.metrics.names = self.names
        self.confusion_matrix = CustomedConfusionMatrix(
            ignore_gt_class=self.ignore_gt_class,
            ignore_FP_pair=self.ignore_FP_pair,
            mapping_gt_pred=self.mapping_gt_pred,
            ori_names=self.ori_names,
            names=self.gt_names, 
            save_dir=self.save_dir, 
            nc=self.num_gt_classes,
            model_nc=len(self.init_names), 
            conf=self.args.conf)
    
    def postprocess(self, preds):
        """
        Apply Non-maximum suppression to prediction outputs.

        Args:
            preds (torch.Tensor): Raw predictions from the model.

        Returns:
            (List[torch.Tensor], List[torch.Tensor]): Processed predictions after NMS.
        """
        return non_max_suppression(
            preds,
            self.args.conf,
            self.args.iou,
            labels=self.lb,
            nc=self.nc,
            multi_label=True,
            agnostic=self.args.single_cls or self.args.agnostic_nms,
            max_det=self.args.max_det,
            end2end=self.end2end,
            rotated=self.args.task == "obb",
        )

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
                    (boxes[:, 2] < W - margin) & (boxes[:, 3] < H - margin)

        return boxes[valid_mask]  # Chỉ giữ các box hợp lệ

    def update_metrics(self, preds, preds1, batch):
        """Metrics."""
        for si, (pred, pred1) in enumerate(zip(preds, preds1)):
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
                        self.confusion_matrix.process_batch(pbatch, detections=None, detections1=None, gt_bboxes=bbox, gt_cls=cls)
                continue

            # Predictions
            if self.args.single_cls:
                pred[:, 5] = 0
            predn = self._prepare_pred(pred, pbatch)
            predn1 = self._prepare_pred(pred1, pbatch)
            # predn = self.filter_boxes_near_border(predn, pbatch['ori_shape'])
            stat["conf"] = predn[:, 4]
            stat["pred_cls"] = predn[:, 5]

            # Evaluate
            if nl:
                stat["tp"] = self._process_batch(predn, bbox, cls)
                if self.args.plots:
                    # self.confusion_matrix.process_batch(predn, bbox, cls)
                    self.confusion_matrix.process_batch(pbatch, predn, predn1, bbox, cls)
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
    
    def print_results(self):
        """Print training/validation set metrics per class."""
        pf = "%22s" + "%11i" * 2 + "%11.3g" * len(self.metrics.keys)  # print format
        LOGGER.info(pf % ("all", self.seen, self.nt_per_class.sum(), *self.metrics.mean_results()))
        if self.nt_per_class.sum() == 0:
            LOGGER.warning(f"WARNING ⚠️ no labels found in {self.args.task} set, can not compute metrics without labels")

        # Print results per class
        if self.args.verbose and not self.training and self.nc > 1 and len(self.stats):
            for i, c in enumerate(self.metrics.ap_class_index):
                LOGGER.info(
                    pf % (self.names[c], self.nt_per_image[c], self.nt_per_class[c], *self.metrics.class_result(i))
                )

        if self.args.plots:
            for normalize in True, False:
                self.confusion_matrix.plot(
                    save_dir=self.save_dir, model_names=self.init_names.values(), label_names=self.names.values(),  
                    normalize=normalize, on_plot=self.on_plot
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

