__version__ = "0.0.7"

from .hed import HEDdetector
from .leres import LeresDetector
from .lineart import LineartDetector
from .lineart_anime import LineartAnimeDetector
from .midas import MidasDetector
from .mlsd import MLSDdetector
from .normalbae import NormalBaeDetector
from .open_pose import OpenposeDetector
from .pidi import PidiNetDetector
from .zoe import ZoeDetector

from .canny import CannyDetector
from .mediapipe_face import MediapipeFaceDetector
from .segment_anything import SamDetector
from .shuffle import ContentShuffleDetector
from .dwpose import DWposeDetector
