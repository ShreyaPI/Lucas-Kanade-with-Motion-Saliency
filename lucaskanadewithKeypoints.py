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

# Create some random colors for drawing motion tracks
color = np.random.randint(0, 255, (1000, 3))

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
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing motion tracks
mask = np.zeros_like(old_frame)

# Initialize a frame counter for periodic key-point recalculation
frame_count = 0
recalc_interval = 30  # Recalculate key points every 20 frames

while True:
    ret, frame = cap.read()
    if not ret:
        print("No more frames to read. Exiting...")
        break

    # Resize the frame for consistent display
    frame = ResizeWithAspectRatio(frame, width=512)

    # Convert to grayscale
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Recalculate key points every 'recalc_interval' frames
    if frame_count % recalc_interval == 0 or p0 is None or len(p0) == 0:
        print("Recalculating key points...")
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
        mask = np.zeros_like(old_frame)  # Reset the mask for clean drawing

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
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)

        # Update previous frame and points
        p0 = good_new.reshape(-1, 1, 2)

    # Overlay the tracks on the frame
    img = cv2.add(frame, mask)

    # Display the output
    cv2.imshow('Resized Video with Optical Flow', img)

    # Increment the frame counter
    frame_count += 1

    # Exit on 'ESC' key
    if cv2.waitKey(25) & 0xFF == 27:
        break

    old_gray = frame_gray.copy()

# Release resources
cap.release()
cv2.destroyAllWindows()
