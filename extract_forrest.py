import pandas as pd

from pliers.stimuli.video import VideoStim
from pliers.converters.video import FrameSamplingConverter
from pliers.extractors.google import GoogleVisionAPIFaceExtractor

dataset_dir = '/Users/alejandro/datasets/forrest/phase2/'

video = os.join(dataset_dir, 'stimuli/movie/fg_av_seg0.mkb')

# Sample video
conv = FrameSamplingConverter(every=15)
derived = conv.transform(video)

ext = GoogleVisionAPIFaceExtractor(discovery_file='/Users/alejandro/bin/forrest-1c48f2c6a8c9.json')

features = [ext.transform(frame) for frame in derived]