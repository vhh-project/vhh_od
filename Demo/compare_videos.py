"""
Takes two videos and puts them side by side making it easier to spot differences.
The videos should have the same width and height.
Written to compare YOLOv3 with YOLOv5.
"""

import cv2
import numpy as np
import argparse
import os

def main(video1_path, video2_path, output_path):
    cap1 = cv2.VideoCapture(video1_path)
    cap2 = cv2.VideoCapture(video2_path)

    out, fourcc = None, None

    video1_info = os.path.split(video1_path)[-1]
    video2_info = os.path.split(video2_path)[-1]

    output_path = os.path.join(output_path, "merged_video.m4v")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('video1', type=str,
                    help='Path to the first video')
    parser.add_argument('video2', type=str,
                    help='Path to the second video')
    parser.add_argument('path_out', metavar='path_out', type=str,
                    help='Path where the output film should be stored')
    args = parser.parse_args()
    main(args.video1, args.video2, args.path_out)

