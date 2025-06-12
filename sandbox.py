from ultralytics import YOLO
import os

model = YOLO("trained model/v2 data/v11s_coco_v2data_PA3_7_classes_quadrant_500ep.pt")  # Load a pretrained YOLOv8 model

model.predict('datasets/v2_malaria_PA3_7_class_quadrant/test/images/051Overlay002_0.jpg',
              save=True, )
