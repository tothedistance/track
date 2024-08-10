import os
import cv2
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../deep_sort_realtime')))
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

embedder = 'clip_ViT-B/32'
embedder = 'clip_RN50x16'
embedder = 'mobilenet'
tracker = DeepSort(max_age=5, embedder=embedder)
img_dir = '/data/MOT/MOT16/train/MOT16-02/img1'
output_video_path = "exp/output_video-%s.avi"%(embedder.replace("/", "-"))
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # Load YOLOv5 detector
 
if __name__ == "__main__":
    # list the images in img_dir and iterate over then
    images = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
    images.sort()  # Sort images to maintain order, assuming filenames allow for correct sorting
    
    # Initialize video writer
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

    for img_path in images:
        frame = cv2.imread(img_path)

        # Run object detector on the frame
        detections = detector(frame)

        # Update tracker with the current detections
        xywhns = detections.xywhn[0].to('cpu')
        xywhns[:, 0] = xywhns[:, 0] * width
        xywhns[:, 1] = xywhns[:, 1] * height
        xywhns[:, 2] = xywhns[:, 2] * width
        xywhns[:, 3] = xywhns[:, 3] * height
        xywhns[:, 0] = xywhns[:, 0] - xywhns[:, 2] / 2
        xywhns[:, 1] = xywhns[:, 1] - xywhns[:, 3] / 2
        measures = [(xyxy, conf, cls) for xyxy, conf, cls in zip(xywhns[:, 0:4], xywhns[:, 4], xywhns[:, 5])]
        tracks = tracker.update_tracks(measures, frame=frame)

        # Draw bounding boxes for each track on the frame
        for track, xywhn in zip(tracks, xywhns):
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            ltrb = track.to_ltrb()
            l, t, r, b = [int(x) for x in ltrb]
            cv2.rectangle(frame, (l, t), (r, b), (255, 0, 0), 2)
            cv2.putText(frame, str(track_id), (l, t-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

        # Write the frame with bounding boxes to the video
        video.write(frame)

    # Release the video writer
    video.release()