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

# Parameters for speed calculation
fps = cap.get(cv2.CAP_PROP_FPS)
scaling_factor = 0.1  # Adjust based on the real-world scene calibration (meters per pixel)
frame_time = 1 / fps  # Time between frames in seconds
vehicle_speeds = {}  # Dictionary to store average speeds of vehicles
speed_history = {}  # Dictionary to store speed history for averaging
vehicle_positions = {}  # Dictionary to track the positions of vehicles

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

    # Recalculate key points every 'recalc_interval' frames or based on saliency changes
    saliency_diff = cv2.absdiff(prev_saliency, motion_saliency)
    saliency_change = np.sum(saliency_diff) / (np.sum(motion_saliency) + 1e-6)

    if frame_count % recalc_interval == 0 or p0 is None or len(p0) == 0 or saliency_change > 0.1:
        print("Recalculating key points...")
        p0 = cv2.goodFeaturesToTrack(frame_gray, mask=motion_saliency, **feature_params)

    # Calculate optical flow
    if p0 is not None:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # Group points by proximity to track vehicles
        new_vehicle_positions = {}
        new_speed_history = {}
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()

            # Calculate displacement in pixels
            displacement = np.sqrt((a - c) ** 2 + (b - d) ** 2)

            # Convert displacement to real-world distance
            real_distance = displacement * scaling_factor

            # Calculate instantaneous speed (m/s)
            speed_mps = real_distance / frame_time

            # Convert speed to km/h
            speed_kmh = speed_mps * 3.6

            if speed_kmh > 5:  # Only track points with speed > 5 km/h
                # Find the closest centroid to associate this point
                assigned = False
                for vehicle_id, centroid in vehicle_positions.items():
                    centroid_x, centroid_y = centroid
                    # Clamp indices to valid range
                    h, w = motion_saliency.shape
                    b = max(0, min(h - 1, int(b)))
                    a = max(0, min(w - 1, int(a)))

                    # Scale distance by saliency
                    distance_threshold = 20 / (1 + motion_saliency[b, a] / 255.0)

                    if np.sqrt((a - centroid_x) ** 2 + (b - centroid_y) ** 2) < distance_threshold:
                        new_vehicle_positions[vehicle_id] = (
                            (vehicle_positions[vehicle_id][0] + a) / 2, 
                            (vehicle_positions[vehicle_id][1] + b) / 2
                        )
                        new_speed_history.setdefault(vehicle_id, speed_history.get(vehicle_id, [])).append(speed_kmh)
                        if len(new_speed_history[vehicle_id]) > 30:
                            new_speed_history[vehicle_id].pop(0)
                        assigned = True
                        break

                # If no existing centroid is close enough, create a new one
                if not assigned:
                    new_id = len(new_vehicle_positions)
                    new_vehicle_positions[new_id] = (a, b)
                    new_speed_history[new_id] = [speed_kmh]

        # Update vehicle positions and speed history
        vehicle_positions = new_vehicle_positions
        speed_history = new_speed_history

        # Draw the tracks and display speeds
        for vehicle_id, centroid in vehicle_positions.items():
            x, y = centroid

            # Calculate average speed
            average_speed_kmh = np.mean(speed_history[vehicle_id])

            # Display average speed on the frame
            cv2.putText(
                frame,
                f"{average_speed_kmh:.2f} km/h",
                (int(x), int(y) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    # Display the output
    saliency_overlay = cv2.addWeighted(frame, 0.7, cv2.cvtColor(motion_saliency, cv2.COLOR_GRAY2BGR), 0.3, 0)
    cv2.imshow('Motion Saliency and Vehicle Speed Tracking', saliency_overlay)

    # Increment the frame counter
    frame_count += 1

    # Exit on 'ESC' key
    if cv2.waitKey(25) & 0xFF == 27:
        break

    old_gray = frame_gray.copy()
    prev_saliency = motion_saliency.copy()

# Release resources
cap.release()
cv2.destroyAllWindows()
