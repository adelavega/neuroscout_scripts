import pandas as pd

from pliers.stimuli.video import VideoStim
from pliers.converters.video import FrameSamplingConverter
from pliers.extractors.google import GoogleVisionAPIFaceExtractor
import os

dataset_dir = '/mnt/c/Users/aid338/Documents/forrest/'

video = os.path.join(dataset_dir, 'stimuli/movie/fg_av_seg0.mkb')

# Sample video
conv = FrameSamplingConverter(hertz=1)
derived = conv.transform(video)

ext = GoogleVisionAPIFaceExtractor(discovery_file='/mnt/c/Users/aid338/Documents/forrest-1c48f2c6a8c9.json')

features = [ext.transform(frame) for frame in derived]