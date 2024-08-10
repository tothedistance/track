import os
import cv2
import numpy as np

def mot_raw_track2video(solution, answer, images, output_path, fps=30):
    frame = cv2.imread(images[0])
    height, width, layers = frame.shape
    out_video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    i = 0
    for index, image in enumerate(images):
        frame = cv2.imread(image)
        while i < len(answer):
            if answer[i][0] == index:
                track = answer[i][1:]
                track_id, bbox = track[0], track[1:5]
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1]+bbox[3]),), (0, 255, 0), 2)
                cv2.putText(frame, str(track_id), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
            else:
                break
            i += 1
        out_video.write(frame)
    out_video.release()

sequence = 'MOT20-01'
sequence_dir = '/data/MOT/MOT20/train/%s/'%sequence
img_dir = sequence_dir + '/img1'
output_path = "./exp/xinCOLOR/output_video-%s.mp4"%sequence
images = [os.path.join(img_dir, img) for img in os.listdir(img_dir) if img.endswith((".png", ".jpg", ".jpeg"))]
images.sort()  # Sort images to maintain order, assuming filenames allow for correct sorting
answer = np.loadtxt('./exp/xinCOLOR/%s.txt'%sequence, delimiter=',')    # Load the answer
solution = np.loadtxt(sequence_dir + '/gt/gt.txt', delimiter=',')   # Load the solution

if __name__ == "__main__":
    mot_raw_track2video(solution, answer, images, output_path)