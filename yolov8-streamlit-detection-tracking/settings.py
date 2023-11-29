from pathlib import Path
import sys

# Get the absolute path of the current file
file_path = Path(__file__).resolve()

# Get the parent directory of the current file
root_path = file_path.parent

# Add the root path to the sys.path list if it is not already there
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Get the relative path of the root directory with respect to the current working directory
ROOT = root_path.relative_to(Path.cwd())

# Sources
IMAGE = 'Image'
VIDEO = 'Video'
WEBCAM = 'Webcam'
RTSP = 'RTSP'
YOUTUBE = 'YouTube'

SOURCES_LIST = [IMAGE, VIDEO, WEBCAM, RTSP, YOUTUBE]

# Images config
IMAGES_DIR = ROOT / 'images'
DEFAULT_IMAGE = IMAGES_DIR / 'roma_00884.png'
DEFAULT_DETECT_IMAGE = IMAGES_DIR / 'roma_00884_detected.png'

# Videos config
VIDEO_DIR = ROOT / 'videos'
VIDEO_1_PATH = VIDEO_DIR / 'ecp-demo-1.mp4'
VIDEO_2_PATH = VIDEO_DIR / 'example-1.mp4'
VIDEOS_DICT = {
    'video_1': VIDEO_1_PATH,
    'video_2': VIDEO_2_PATH
}

# ML Model config
MODEL_DIR = ROOT / 'weights'
BASE_MODEL = MODEL_DIR / 'yolov8n.pt'
ECP_MODEL = MODEL_DIR / 'ecp_best.pt'
CALTECH_MODEL = MODEL_DIR / 'caltech_best.pt'

# Webcam
WEBCAM_PATH = 0
