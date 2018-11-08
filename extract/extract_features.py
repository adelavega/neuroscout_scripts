''' Extracts features from a video stimulus'''

import sys
import numpy as np
import pandas as pd

from pliers.export import to_long_format
from pliers.extractors import (ClarifaiAPIExtractor,
                               VibranceExtractor,
                               STFTAudioExtractor,
                               merge_results)
from pliers.filters import FrameSamplingFilter
from pliers.stimuli import VideoStim

TR = 2.00000061

def extract_image_labels(video, save_frames=False):
    # Sample frames at TR
    frame_sampling_filter = FrameSamplingFilter(every=int(TR*video.fps))
    sampled_video = frame_sampling_filter.transform(video)

    # Use a Vision API to extract object labels
    ext = ClarifaiAPIExtractor()
    results = ext.transform(sampled_video)
    res = merge_results(results, metadata=False)
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/clarifai_object_events_all.csv')

def extract_vibrance(video):
    frame_sampling_filter = FrameSamplingFilter(every=5)
    sampled_video = frame_sampling_filter.transform(video)
    ext = VibranceExtractor()
    res = ext.transform(sampled_video)
    res = merge_results(res, metadata=False, flatten_columns=True)
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/visual_vibrance_events.csv')

def extract_audio_frequency_features(video):
    ext = STFTAudioExtractor(hop_size=1, freq_bins=[(60, 250)])
    res = ext.transform(video)
    res = res.to_df()
    res = to_long_format(res)
    res.rename(columns={'value': 'modulation', 'feature': 'trial_type'}, inplace=True)
    res.to_csv('events/audio_frequency_60250.csv')

if __name__ == '__main__':
    if len(sys.argv) != 1:
        print('Usage: python extract_features.py')
        sys.exit(0)

    video = VideoStim('Merlin.mp4')
    extract_image_labels(video)
    print('Done with label extraction')
    # extract_visual_semantics('events/raw_visual_events.csv')
    # print('Done with visual semantics extraction')
    print('Done with visual object extraction')
    # extract_audio_semantics(parse_p2fa('stims/transcription/Merlin_trimmed.TextGrid'))
    # print('Done with audio semantics extraction')
    # extract_audio_energy(video)
    # extract_speech()
    extract_vibrance(video)
    extract_audio_frequency_features(video)
    # extract_faces(video)
