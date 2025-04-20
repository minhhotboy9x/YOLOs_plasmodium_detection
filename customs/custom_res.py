
import torch

from copy import deepcopy
from pathlib import Path

from ultralytics.utils import LOGGER
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils.plotting import Annotator, colors
from ultralytics.utils.plotting import Annotator, colors, save_one_box
from ultralytics.utils.files import increment_path



class CustomedResults(Results):
    def __init__(
        self, orig_img, path, names, boxes=None, masks=None, probs=None, keypoints=None, obb=None, speed=None, 
        gt_labels=None
    ) -> None:
        super().__init__(orig_img, path, names, boxes, masks, probs, keypoints, obb, speed)
        self.gt_labels = gt_labels
    
    def plot(
        self,
        conf=True,
        line_width=None,
        font_size=None,
        font="Arial.ttf",
        pil=False,
        img=None,
        im_gpu=None,
        kpt_radius=5,
        kpt_line=True,
        labels=True,
        boxes=True,
        masks=True,
        probs=True,
        show=False,
        save=False,
        filename=None,
        color_mode="class",
    ):
        """
        Plots detection results on an input RGB image.

        Args:
            conf (bool): Whether to plot detection confidence scores.
            line_width (float | None): Line width of bounding boxes. If None, scaled to image size.
            font_size (float | None): Font size for text. If None, scaled to image size.
            font (str): Font to use for text.
            pil (bool): Whether to return the image as a PIL Image.
            img (np.ndarray | None): Image to plot on. If None, uses original image.
            im_gpu (torch.Tensor | None): Normalized image on GPU for faster mask plotting.
            kpt_radius (int): Radius of drawn keypoints.
            kpt_line (bool): Whether to draw lines connecting keypoints.
            labels (bool): Whether to plot labels of bounding boxes.
            boxes (bool): Whether to plot bounding boxes.
            masks (bool): Whether to plot masks.
            probs (bool): Whether to plot classification probabilities.
            show (bool): Whether to display the annotated image.
            save (bool): Whether to save the annotated image.
            filename (str | None): Filename to save image if save is True.
            color_mode (bool): Specify the color mode, e.g., 'instance' or 'class'. Default to 'class'.

        Returns:
            (np.ndarray): Annotated image as a numpy array.

        Examples:
            >>> results = model("image.jpg")
            >>> for result in results:
            ...     im = result.plot()
            ...     im.show()
        """
        assert color_mode in {"instance", "class"}, f"Expected color_mode='instance' or 'class', not {color_mode}."
        if img is None and isinstance(self.orig_img, torch.Tensor):
            img = (self.orig_img[0].detach().permute(1, 2, 0).contiguous() * 255).to(torch.uint8).cpu().numpy()

        names = self.names
        is_obb = self.obb is not None
        pred_boxes, show_boxes = self.obb if is_obb else self.boxes, boxes
        pred_masks, show_masks = self.masks, masks
        pred_probs, show_probs = self.probs, probs
        annotator = Annotator(
            deepcopy(self.orig_img if img is None else img),
            line_width,
            font_size,
            font,
            pil or (pred_probs is not None and show_probs),  # Classify tasks default to pil=True
            example=names,
        )

        # Plot Segment results
        if pred_masks and show_masks:
            if im_gpu is None:
                img = LetterBox(pred_masks.shape[1:])(image=annotator.result())
                im_gpu = (
                    torch.as_tensor(img, dtype=torch.float16, device=pred_masks.data.device)
                    .permute(2, 0, 1)
                    .flip(0)
                    .contiguous()
                    / 255
                )
            idx = (
                pred_boxes.id
                if pred_boxes.id is not None and color_mode == "instance"
                else pred_boxes.cls
                if pred_boxes and color_mode == "class"
                else reversed(range(len(pred_masks)))
            )
            annotator.masks(pred_masks.data, colors=[colors(x, True) for x in idx], im_gpu=im_gpu)

        # Plot Detect results
        if pred_boxes is not None and show_boxes:
            gt_labels = list(reversed(self.gt_labels)) if self.gt_labels is not None else None
            for i, d in enumerate(reversed(pred_boxes)):
                c, conf, id = int(d.cls), float(d.conf) if conf else None, None if d.id is None else int(d.id.item())
                name = (f"{i} ") + ("" if id is None else f"id:{id} ") + names[c]
                if gt_labels is not None:
                    name += f" | GT: {gt_labels[i]}"
                label = (f"{name} {conf:.2f}" if conf else name) if labels else None
                box = d.xyxyxyxy.reshape(-1, 4, 2).squeeze() if is_obb else d.xyxy.squeeze()
                annotator.box_label(
                    box,
                    label,
                    color=colors(
                        c
                        if color_mode == "class"
                        else id
                        if id is not None
                        else i
                        if color_mode == "instance"
                        else None,
                        True,
                    ),
                    rotated=is_obb,
                )

        # Plot Classify results
        if pred_probs is not None and show_probs:
            text = ",\n".join(f"{names[j] if names else j} {pred_probs.data[j]:.2f}" for j in pred_probs.top5)
            x = round(self.orig_shape[0] * 0.03)
            annotator.text([x, x], text, txt_color=(255, 255, 255))  # TODO: allow setting colors

        # Plot Pose results
        if self.keypoints is not None:
            for i, k in enumerate(reversed(self.keypoints.data)):
                annotator.kpts(
                    k,
                    self.orig_shape,
                    radius=kpt_radius,
                    kpt_line=kpt_line,
                    kpt_color=colors(i, True) if color_mode == "instance" else None,
                )

        # Show results
        if show:
            annotator.show(self.path)

        # Save results
        if save:
            annotator.save(filename)

        return annotator.result()
    
    def save_crop_and_box(self, save_dir, file_name=Path("im.jpg")):
        """
        Saves cropped detection images to specified directory.

        This method saves cropped images of detected objects to a specified directory. Each crop is saved in a
        subdirectory named after the object's class, with the filename based on the input file_name.

        Args:
            save_dir (str | Path): Directory path where cropped images will be saved.
            file_name (str | Path): Base filename for the saved cropped images. Default is Path("im.jpg").

        Notes:
            - This method does not support Classify or Oriented Bounding Box (OBB) tasks.
            - Crops are saved as 'save_dir/class_name/file_name.jpg'.
            - The method will create necessary subdirectories if they don't exist.
            - Original image is copied before cropping to avoid modifying the original.

        Examples:
            >>> results = model("path/to/image.jpg")
            >>> for result in results:
            ...     result.save_crop(save_dir="path/to/crops", file_name="detection")
        """
        if self.probs is not None:
            LOGGER.warning("WARNING ⚠️ Classify task do not support `save_crop`.")
            return
        if self.obb is not None:
            LOGGER.warning("WARNING ⚠️ OBB task do not support `save_crop`.")
            return
        for d in self.boxes:
            save_one_box(
                d.xyxy,
                self.orig_img.copy(),
                file=Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.jpg",
                BGR=True,
            )
            box_file = Path(save_dir) / self.names[int(d.cls)] / f"{Path(file_name)}.txt"
            box_file = increment_path(box_file)
            xyxy = d.xyxy.cpu().numpy().tolist()
            with open(box_file, "w") as f:
                for coord in xyxy[0]:
                    f.write(str(coord) + " ")

    def save_pred_gt_txt(self, txt_file, save_conf=False):
        """
        Save detection results to a text file.

        Args:
            txt_file (str | Path): Path to the output text file.
            save_conf (bool): Whether to include confidence scores in the output.

        Returns:
            (str): Path to the saved text file.
        """

        is_obb = self.obb is not None
        boxes = self.obb if is_obb else self.boxes
        gt_labels = self.gt_labels
        names = self.names
        texts = []
        if boxes:
            gt_labels = list(reversed(self.gt_labels))
            for j, d in enumerate(reversed(boxes)):
                c, conf, id = int(d.cls.item()), float(d.conf.item()), None if d.id is None else int(d.id.item())
                line = (j, ) + (names[c],) + (f"| GT: {gt_labels[j]}" if gt_labels else (),) + tuple(d.xyxyxyxyn.view(-1).tolist() if is_obb else d.xywhn.view(-1).tolist())
                line += (conf,) * save_conf + (() if id is None else (id,))
                texts.append(" ".join(f"{x:.7g}" if isinstance(x, (int, float)) else str(x) for x in line))

        if texts:
            Path(txt_file).parent.mkdir(parents=True, exist_ok=True)  # make directory
            with open(txt_file, "a") as f:
                f.writelines(text + "\n" for text in texts)