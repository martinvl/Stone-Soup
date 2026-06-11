import datetime
import heapq
from warnings import warn

from ..base import Property
from ..buffered_generator import BufferedGenerator
from .base import DetectionFeeder, GroundTruthFeeder
from ..types.detection import DetectionSet


def _new_data_buffer(states):
    detector_context = getattr(states, 'detector_context', None)
    if detector_context is None:
        return set(states)
    return DetectionSet(states, detector_context=detector_context)


def _update_data_buffer(data_buffer, states):
    detector_context = getattr(data_buffer, 'detector_context', None)
    new_detector_context = getattr(states, 'detector_context', None)
    values = set(data_buffer)
    values.update(states)
    if detector_context is not None and (
            new_detector_context is None or new_detector_context == detector_context):
        return DetectionSet(values, detector_context=detector_context)
    if detector_context is None and new_detector_context is not None:
        return DetectionSet(values, detector_context=new_detector_context)
    return values


class TimeBufferedFeeder(DetectionFeeder, GroundTruthFeeder):
    """Buffer data so it can be yielded in time order.

    Any "old" data (where the time is earlier than the head of the
    buffer) shall be dropped, producing a :class:`UserWarning`.
    """
    buffer_size: int = Property(default=1000, doc="Max size of buffer")

    @BufferedGenerator.generator_method
    def data_gen(self):
        time_data_buffer = []

        for time_data in self.reader:
            # Drop "old" detections
            if len(time_data_buffer) >= self.buffer_size and \
                    time_data < time_data_buffer[0]:
                warn('"Old" detection dropped')
                continue

            # Yield oldest when buffer full
            if len(time_data_buffer) >= self.buffer_size:
                yield heapq.heappushpop(time_data_buffer, time_data)
            else:
                # Else just insert
                heapq.heappush(time_data_buffer, time_data)

        # No more new data: yield remaining buffer
        while time_data_buffer:
            yield heapq.heappop(time_data_buffer)


class TimeSyncFeeder(DetectionFeeder, GroundTruthFeeder):
    """Synchronise the data into selected time window.

    Assumes that states from :attr:`Reader` are in time order. The
    :class:`~.TimeBufferedFeeder` can be used in conjunction to ensure this is
    the case.
    """

    time_window: datetime.timedelta = Property(
        default=datetime.timedelta(seconds=1),
        doc="Time window to group detections")

    @BufferedGenerator.generator_method
    def data_gen(self):
        data_iter = iter(self.reader)
        prev_time, states = next(data_iter)
        data_buffer = _new_data_buffer(states)

        prev_time -= self.time_window

        for time, states in data_iter:
            if time > prev_time + self.time_window:
                yield prev_time + self.time_window, data_buffer

                # Increment time and yield empty data set until latest
                # state within time window.
                prev_time += self.time_window
                while time > prev_time + self.time_window:
                    prev_time += self.time_window
                    yield prev_time, set()

                data_buffer = _new_data_buffer(states)
            else:
                data_buffer = _update_data_buffer(data_buffer, states)

        # No more new states: yield remaining buffer
        yield prev_time + self.time_window, data_buffer
