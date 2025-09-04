import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class SingleVideo(Dataset):
    def __init__(self, work_path="AVA_cache/single_video/"):
        """
        Args:
            videos_path (string): Directory with all the videos.
        Self:
            video_info: 
                video_path -> source video path
                others -> other video dataset information
            work_path: directory to save the processed video frames
        """        
        self.work_path = work_path
    
    def get_video(self, videos_path):        
        source_path = videos_path
        work_path = os.path.join(self.work_path)
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(source_path, work_path)