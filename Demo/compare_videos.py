"""
Takes two videos and puts them side by side making it easier to spot differences.
The videos should have the same width and height.
Written to compare YOLOv3 with YOLOv5.
"""

import cv2
import numpy as np

video1_path = "/data/share/fjogl/vhh_od_dev_results/raw_results/vis/8220_results_yolov3.avi"
video2_path = "/data/share/fjogl/vhh_od_dev_results/raw_results/vis/8220_results_yolov3_new.avi"

video1_info = "YOLOv3"
video2_info = "YOLOv3_retrained"

output_path = "/data/share/fjogl/vhh_od_dev_results/raw_results/vis/output.mp4"

def main():
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    out, fourcc = None, None

    # Merge videos
    for iter in __import__("itertools").count():
        print("\rWritten {0} frames".format(iter), end="")

        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        # Stop if any of the videos have ended
        if not ret1 or not ret2:
            break

        frame1 = cv2.resize(frame1, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)
        frame2 = cv2.resize(frame2, None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_CUBIC)

        cv2.putText(frame1, video1_info, (20, frame1.shape[0]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        cv2.putText(frame2, video2_info, (20, frame1.shape[0]- 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)

        combined_frame = np.hstack([frame1, frame2])

        if out is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, 24, (combined_frame.shape[1], combined_frame.shape[0]))

        out.write(combined_frame)

    out.release()
    



if __name__ == "__main__":
    main()

