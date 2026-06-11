import datetime

import numpy as np
import pytest

from ..probability import PDAHypothesiser
from ...types.detection import Detection, MissedDetection
from ...types.detector_context import SimpleDetectorContext
from ...types.numeric import Probability
from ...types.state import GaussianState
from ...types.track import Track


@pytest.mark.parametrize('normalise', [True, False])
def test_pda(predictor, updater, normalise):

    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[8]]))
    detections = {detection1, detection2}

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9, prob_gate=0.99, normalise=normalise)

    mulltihypothesis = \
        hypothesiser.hypothesise(track, detections, timestamp)

    # There are 2 weighted hypotheses - Detections 1 and MissedDetection
    # Detection 2 is not a "valid measurement" and gated out
    assert len(mulltihypothesis) == 2

    # Each hypothesis has a probability/weight attribute
    assert all(hypothesis.probability >= 0 and isinstance(hypothesis.probability, Probability)
               for hypothesis in mulltihypothesis)

    #  Detection 1 and MissedDetection are present
    assert detection1 in {hypothesis.measurement for hypothesis in mulltihypothesis}
    assert any(isinstance(hypothesis.measurement, MissedDetection)
               for hypothesis in mulltihypothesis)

    if normalise:
        assert float(sum(hypothesis.weight for hypothesis in mulltihypothesis)) == pytest.approx(1)
    else:
        assert float(sum(hypothesis.weight for hypothesis in mulltihypothesis)) != pytest.approx(1)

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   clutter_spatial_density=1.2e-2,
                                   prob_detect=0.9, prob_gate=0.99,
                                   include_all=True, normalise=normalise)

    mulltihypothesis = \
        hypothesiser.hypothesise(track, detections, timestamp)

    # There are 3 weighted hypotheses - Detections 1 and 2, MissedDetection
    assert len(mulltihypothesis) == 3

    # Each hypothesis has a probability/weight attribute
    assert all(hypothesis.probability >= 0 and isinstance(hypothesis.probability, Probability)
               for hypothesis in mulltihypothesis)

    #  Detection 1, 2 and MissedDetection are present
    assert {detection1, detection2} < {hypothesis.measurement for hypothesis in mulltihypothesis}
    assert any(isinstance(hypothesis.measurement, MissedDetection)
               for hypothesis in mulltihypothesis)

    if normalise:
        assert float(sum(hypothesis.weight for hypothesis in mulltihypothesis)) == pytest.approx(1)
    else:
        assert float(sum(hypothesis.weight for hypothesis in mulltihypothesis)) != pytest.approx(1)

    hypothesiser = PDAHypothesiser(predictor, updater,
                                   prob_detect=0.9, prob_gate=0.99, normalise=normalise)

    mulltihypothesis = \
        hypothesiser.hypothesise(track, detections, timestamp)

    # There are 2 weighted hypotheses - Detections 1 and MissedDetection
    # Detection 2 is not a "valid measurement" and gated out
    assert len(mulltihypothesis) == 2

    # Each hypothesis has a probability/weight attribute
    assert all(hypothesis.probability >= 0 and isinstance(hypothesis.probability, Probability)
               for hypothesis in mulltihypothesis)

    #  Detection 1 and MissedDetection are present
    assert detection1 in {hypothesis.measurement for hypothesis in mulltihypothesis}
    assert any(isinstance(hypothesis.measurement, MissedDetection)
               for hypothesis in mulltihypothesis)

    if normalise:
        assert float(sum(hypothesis.weight for hypothesis in mulltihypothesis)) == pytest.approx(1)
    else:
        assert float(sum(hypothesis.weight for hypothesis in mulltihypothesis)) != pytest.approx(1)


def test_pda_detector_context(predictor, updater):
    timestamp = datetime.datetime.now()
    track = Track([GaussianState(np.array([[0]]), np.array([[1]]), timestamp)])
    detection1 = Detection(np.array([[2]]))
    detection2 = Detection(np.array([[8]]))
    detections = {detection1, detection2}
    detector_context = SimpleDetectorContext(
        prob_detection=0.9,
        clutter_spatial_density=1.2e-2)

    scalar_hypothesiser = PDAHypothesiser(
        predictor, updater, clutter_spatial_density=1.2e-2,
        prob_detect=0.9, prob_gate=0.99, normalise=False)
    context_hypothesiser = PDAHypothesiser(
        predictor, updater, prob_gate=0.99, normalise=False)

    scalar_hypotheses = scalar_hypothesiser.hypothesise(track, detections, timestamp)
    context_hypotheses = context_hypothesiser.hypothesise(
        track, detections, timestamp, detector_context=detector_context)

    scalar_probabilities = {
        None if isinstance(hypothesis.measurement, MissedDetection) else hypothesis.measurement:
        hypothesis.probability for hypothesis in scalar_hypotheses}
    context_probabilities = {
        None if isinstance(hypothesis.measurement, MissedDetection) else hypothesis.measurement:
        hypothesis.probability for hypothesis in context_hypotheses}
    assert scalar_probabilities == context_probabilities


def test_invalid_pda_arguments(predictor, updater):
    with pytest.raises(ValueError):
        PDAHypothesiser(predictor, updater,
                        prob_detect=0.9, prob_gate=0.99, include_all=True)
