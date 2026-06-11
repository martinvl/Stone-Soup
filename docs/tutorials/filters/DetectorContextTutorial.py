#!/usr/bin/env python

"""
=========================
Detector context tutorial
=========================
"""

# %%
# Some probabilistic trackers use detection probability and clutter density directly in their
# hypothesis weights. A detector context lets these quantities depend on the current hypothesis or
# detection while preserving the existing scalar configuration path.

import datetime

import numpy as np

from stonesoup.models.measurement.linear import LinearGaussian
from stonesoup.types.detection import Detection, DetectionSet
from stonesoup.types.detector_context import SimpleDetectorContext
from stonesoup.types.hypothesis import SingleHypothesis
from stonesoup.types.multihypothesis import MultipleHypothesis
from stonesoup.types.state import TaggedWeightedGaussianState
from stonesoup.updater.kalman import KalmanUpdater
from stonesoup.updater.pointprocess import PHDUpdater


timestamp = datetime.datetime.now()
measurement_model = LinearGaussian(
    ndim_state=2,
    mapping=[0],
    noise_covar=np.array([[1.0]]))
kalman_updater = KalmanUpdater(measurement_model=measurement_model)

prediction = TaggedWeightedGaussianState(
    state_vector=np.array([[60.0], [1.0]]),
    covar=np.diag([4.0, 1.0]),
    weight=0.8,
    tag="track-1",
    timestamp=timestamp)
detection = Detection(
    state_vector=np.array([[61.0]]),
    timestamp=timestamp,
    measurement_model=measurement_model)

hypotheses = [
    MultipleHypothesis([SingleHypothesis(prediction, detection)]),
    MultipleHypothesis([SingleHypothesis(prediction, None)]),
]

# %%
# Scalar configuration remains valid. Internally this is equivalent to a constant detector
# context.

updater = PHDUpdater(
    updater=kalman_updater,
    prob_detection=0.9,
    clutter_spatial_density=1e-3)
scalar_update = updater.update(hypotheses)

# %%
# A detector context can instead evaluate the same quantities from the hypothesis and detection.
# When supplied by a reader, compatible trackers pass the context from the detection set to
# tracking components using the ``detector_context`` keyword.

detector_context = SimpleDetectorContext(
    prob_detection=lambda hypothesis:
        0.9 if hypothesis.prediction.state_vector[0, 0] < 50 else 0.2,
    clutter_spatial_density=lambda detection_:
        1e-3 if detection_.state_vector[0, 0] < 50 else 1e-2)
detections = DetectionSet({detection}, detector_context=detector_context)
context_update = updater.update(hypotheses, detector_context=detections.detector_context)

print("Scalar weights:", [component.weight for component in scalar_update])
print("Context weights:", [component.weight for component in context_update])
