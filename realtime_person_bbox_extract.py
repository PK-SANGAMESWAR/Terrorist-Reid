# loading dependencies
from ultralytics import YOLO
import cv2
import numpy as np
import torch
import os
from get_person_crop_embedding import get_osnet_1x_embedding

# enable GPU access
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# helper function to set vid. capture params
def capture_setter(vid_path):
    cap = cv2.VideoCapture(vid_path)
    frame_width = 1280
    frame_height = 720
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
    return cap

model = YOLO('yolov8n.pt')
model.to(device)

def vid_to_hot_cache(vid):
    hot_cache = []
    cap = capture_setter(vid)
    save_dir = 'hot_bbox_cache/'
    while cap.isOpened():
        idx = 1
        ret, frame = cap.read()
        if frame is None or cv2.waitKey(1) & 0xFF == 27:
            break
        else:
            person_bboxes = model.predict(
                frame,
                imgsz=640,
                conf=0.45,          # confidence threshold
                iou=0.7,            # NMS IoU threshold
                classes=[0],        # only person class
                save=False,         # do not save/visualize internally
                verbose=False
                )
    
            annotated_frame = person_bboxes[0].plot()
            cv2.imshow('Video Inference', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            result = person_bboxes[0]
            h, w = frame.shape[:2]
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box
                    # clip coords
                    x1 = int(np.clip(x1, 0, w - 1))
                    y1 = int(np.clip(y1, 0, h - 1))
                    x2 = int(np.clip(x2, 0, w - 1))
                    y2 = int(np.clip(y2, 0, h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    else:
                        crop = frame[y1:y2, x1:x2]
                        returned_osnet_1x_embedding = get_osnet_1x_embedding(crop) # generate embedding
                cv2.imshow('Video Inference', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cap.release()
    cv2.destroyAllWindows()