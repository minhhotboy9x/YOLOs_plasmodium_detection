import os
import numpy as np




def save_full_det_gt(path, preds, gt_labels, det_names):
    """
    Save the full detection ground truth and predictions to a file.

    Args:
        path (str): The file path where the data will be saved.
        pred (list): List of predicted bounding boxes and labels.
        gt (list): List of ground truth bounding boxes and labels.
        det_names (list): List of detection names.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        for i, (pred, gt_lbl) in enumerate(zip(preds, gt_labels)):
            pred = pred.cpu().numpy()
            score_strs = [f"{name} {s:.2f}" for name, s in zip(det_names, pred[4:])]
            line = f"{i}: {pred[0]} {pred[1]} {pred[2]} {pred[3]} | " + " | ".join(score_strs) + f" | GT:{gt_lbl}" + "\n"
            f.write(line)