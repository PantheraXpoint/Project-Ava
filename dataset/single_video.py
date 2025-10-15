import json
import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from video_utils import VideoRepresentation
from typing import Union

class SingleVideo(Dataset):
    def __init__(self, work_path="AVA_cache/single_video/", video_path="datas/single_video/video.mp4"):
        """
        Args:
            videos_path (string): Directory with all the videos.
        Self:
            video_info: 
                video_path -> source video path
                others -> other video dataset information
            work_path: directory to save the processed video frames
        """        
        self.work_path = work_path + os.path.basename(video_path)
        self.video_path = video_path
    
    def get_video(self):        
        source_path = self.video_path
        work_path = os.path.join(self.work_path)
        if not os.path.exists(work_path):
            os.makedirs(work_path)
        
        return VideoRepresentation(source_path, work_path)