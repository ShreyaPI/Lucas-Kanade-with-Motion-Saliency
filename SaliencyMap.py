import cv2
import numpy as np
import sys

def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

# Check if the input video file name is provided
if len(sys.argv) < 2:
    print("Usage: python script.py <input_video_file>")
    sys.exit(1)

video_file = sys.argv[1]

# Open the video file
cap = cv2.VideoCapture(video_file)

if not cap.isOpened():
    print(f"Error: Cannot open video file '{video_file}'.")
    sys.exit(1)

# Parameters for corner detection
feature_params = dict(maxCorners=200, qualityLevel=0.2, minDistance=5, blockSize=5)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
)

# Read the first frame
ret, old_frame = cap.read()
if not ret:
    print("Cannot read video. Exiting...")
    cap.release()
    sys.exit(1)

# Resize the first frame for consistent display
old_frame = ResizeWithAspectRatio(old_frame, width=512)

# Convert to grayscale and detect good features to track
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Initialize saliency variables
prev_saliency = np.zeros_like(old_gray)
accumulated_saliency = np.zeros_like(old_gray, dtype=np.float32)

# Initialize key points
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Initialize a frame counter for periodic key-point recalculation
frame_count = 0
recalc_interval = 30  # Recalculate key points every 30 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read. Exiting...")
        break

    # Resize the frame for consistent display
    frame = ResizeWithAspectRatio(frame, width=512)

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Generate a motion saliency map using frame differencing
    motion_saliency = cv2.absdiff(old_gray, frame_gray)
    _, motion_saliency = cv2.threshold(motion_saliency, 30, 255, cv2.THRESH_BINARY)

    # Accumulate saliency map over time for visualization
    accumulated_saliency = cv2.addWeighted(accumulated_saliency, 0.9, motion_saliency.astype(np.float32), 0.1, 0)

    # Recalculate key points every 'recalc_interval' frames
    if frame_count % recalc_interval == 0 or p0 is None or len(p0) == 0:
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=motion_saliency, **feature_params)

    # Calculate optical flow
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            cv2.line(frame, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
            cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)

    # Display the output
    saliency_overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(motion_saliency, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow('Motion Saliency and Vehicle Speed Tracking', saliency_overlay)

    # Show the separate saliency map
    cv2.imshow('Raw Saliency Map', motion_saliency)

    # Increment the frame counter
    frame_count += 1

    # Exit on 'ESC' key
    if cv2.waitKey(25) & 0xFF == 27:
        break

    old_gray = frame_gray.copy()

# Release resources
cap.release()
cv2.destroyAllWindows()
