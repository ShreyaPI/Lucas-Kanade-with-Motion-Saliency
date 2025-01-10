### Motion Saliency and Vehicle Speed Tracking with Lucas-Kanade Optical Flow


#### **Overview**
This script implements vehicle speed tracking in a video using a combination of motion saliency maps and the Lucas-Kanade optical flow algorithm. It detects and tracks vehicles by analyzing motion patterns and calculates their speeds in real-world units (km/h). The output includes a visualization of motion saliency overlaid on the video, along with the tracked vehicle speeds displayed near their positions.


#### **Features**
- **Motion Saliency Detection**: Highlights regions of motion to prioritize tracking.
- **Lucas-Kanade Optical Flow**: Tracks points on moving vehicles across frames.
- **Real-World Speed Estimation**: Converts pixel-based displacement into speed (km/h) using a scaling factor.
- **Dynamic Key Point Recalculation**: Periodically or adaptively recalculates key points for robust tracking.
- **Visual Overlays**: Displays motion saliency and tracked vehicle speeds on the video.


#### **Requirements**
- Python 3.x
- OpenCV 4.x
- NumPy

#### **Installation**
1. Clone this repository or download the script.
2. Install the required libraries:
   ```bash
   pip install opencv-python numpy
   ```

#### **Usage**
1. Place the video file you want to analyze in the same directory as the script.
2. Run the script with the video file as an argument:
   ```bash
   python script.py <input_video_file>
   ```
   Replace `<input_video_file>` with the path to your video file.

3. Use the `ESC` key to exit the visualization.

#### **Parameters**
- **Feature Detection Parameters**:
  - `maxCorners=200`: Maximum number of points to detect.
  - `qualityLevel=0.2`: Minimum quality of features.
  - `minDistance=5`: Minimum distance between detected points.
  - `blockSize=5`: Size of the detection block.

- **Lucas-Kanade Optical Flow Parameters**:
  - `winSize=(15, 15)`: Size of the window for optical flow computation.
  - `maxLevel=2`: Maximum pyramid levels for optical flow computation.
  - `criteria`: Stopping criteria for optimization.

- **Real-World Speed Estimation**:
  - `scaling_factor=0.1`: Conversion factor from pixels to meters (adjust based on calibration).
  - `fps`: Frames per second of the video (automatically determined).

- **Key Point Recalculation**:
  - `recalc_interval=30`: Recalculate key points every 30 frames.

#### **Output**
- Displays the video with:
  - **Motion Saliency Overlay**: Highlights regions with detected motion.
  - **Speed Annotations**: Displays average speed (in km/h) near tracked vehicles.

#### **Notes**
- **Real-World Scaling**: 
  - Update the `scaling_factor` to match your scene's real-world calibration (e.g., meters per pixel).
  - The `fps` is automatically derived from the video but should correspond to the actual frame rate of the input.

- **Performance**:
  - The script dynamically recalculates key points based on saliency changes or frame intervals to ensure robust tracking.
  - Designed for real-time processing but may require adjustments for high-resolution videos.
