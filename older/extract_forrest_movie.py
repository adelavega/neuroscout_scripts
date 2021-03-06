import pandas as pd

from pliers.stimuli.video import VideoStim
from pliers.converters.video import FrameSamplingConverter
from pliers.extractors.google import GoogleVisionAPIFaceExtractor
from os.path import join, isfile
from glob import glob
import re

hertz = 2
videos = [VideoStim(f) for f in sorted(glob('D:\neuroscout\datasets\hcp\movie_stimulus\Post_20140821_version\*'))]
replace = False

# Sampler and extractor
conv = FrameSamplingConverter(hertz=hertz)
ext = GoogleVisionAPIFaceExtractor(discovery_file='C:/Users/aid338/Documents/forrest-1c48f2c6a8c9.json')

for video in videos:
    pattern = re.findall('7T_MOVIE([0-9]*)_CC1_v2', video.filename)
    if pattern:
        segment = int(pattern[0])
        print pattern
        out_file = (join('hcp_extract_results', 'clip{}_googleface_{}hz.csv').format(segment, hertz))
        if replace or (not isfile(out_file)):
            derived = conv.transform(video)
            features = [ext.transform(frame) for frame in derived]

            feats_df = pd.concat([res.to_df() for res in features])
            feats_df.to_csv(out_file)
