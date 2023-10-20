import pathlib

import numpy
import pandas
import pyinterp
import xarray

from . import orf

#: Orbit Repetition File
ORF = pathlib.Path(__file__).parent / 'SWOT_ORF.txt'

#: Orbit File
DATASET = pathlib.Path(__file__).parent / 'SWOT_orbit.nc'

#: Number of passes per cycle
PASSES_PER_CYCLE = 584


def get_cycle_duration(dataset: xarray.Dataset) -> numpy.timedelta64:
    """Return the duration of a cycle.

    Args:
        dataset: Dataset containing the orbit file.

    Returns:
        Duration of a cycle.
    """
    start_time = dataset.start_time[0].values
    end_time = dataset.end_time[-1].values
    return end_time - start_time


def calculate_cycle_axis(
        cycle_duration: numpy.timedelta64) -> pyinterp.TemporalAxis:
    """Calculate the cycle axis.

    Args:
        cycle_duration: Duration of a cycle

    Returns:
        Temporal axis of the cycle.
    """
    cycles = orf.load(ORF)

    cycle_first_measurement = numpy.full((200, ),
                                         numpy.datetime64('NAT'),
                                         dtype='M8[ns]')
    for item in sorted(cycles):
        cycle_first_measurement[item - 1] = cycles[item]
    undefined = numpy.isnat(cycle_first_measurement)
    cycle_first_measurement[undefined] = numpy.full(
        (undefined.sum(), ), cycle_duration, dtype='m8[ns]') * numpy.arange(
            1, 1 + undefined.sum()) + cycles[item]
    return pyinterp.TemporalAxis(cycle_first_measurement)


def get_selected_passes(
        date: numpy.datetime64,
        search_duration: numpy.timedelta64 | None = None
) -> pyinterp.TemporalAxis:
    """Return the selected passes.

    Args:
        date: Date of the first pass.
        search_duration: Duration of the search.

    Returns:
        Temporal axis of the selected passes.
    """
    with xarray.open_dataset(DATASET.resolve()) as ds:
        cycle_duration = get_cycle_duration(ds)
        search_duration = search_duration or cycle_duration
        axis = calculate_cycle_axis(cycle_duration)
        dates = numpy.array([date, date + search_duration])
        indices = axis.find_indexes(dates).ravel()
        cycle_numbers = numpy.repeat(
            numpy.arange(indices[0], indices[-1]) + 1, PASSES_PER_CYCLE)
        axis_slice = axis[indices[0]:indices[-1] + 1]
        pass_numbers = numpy.tile(numpy.arange(1, PASSES_PER_CYCLE + 1),
                                  indices[-1] - indices[0])
        dates_of_selected_passes = numpy.vstack(
            (ds.start_time.values, ) * len(axis_slice)).T + axis_slice
        dates_of_selected_passes = dates_of_selected_passes.T.ravel()
        selected_passes = pyinterp.TemporalAxis(
            dates_of_selected_passes).find_indexes(dates).ravel()
        size = selected_passes[-1] - selected_passes[0]

        result: numpy.ndarray = numpy.ndarray(
            (size, ),
            dtype=[('cycle_number', numpy.uint16),
                   ('pass_number', numpy.uint16),
                   ('first_measurement', 'M8[ns]'),
                   ('last_measurement', 'M8[ns]')])
        axis_slice = slice(selected_passes[0], selected_passes[-1])
        result['cycle_number'] = cycle_numbers[axis_slice]
        result['pass_number'] = pass_numbers[axis_slice]
        result['first_measurement'] = dates_of_selected_passes[axis_slice]
        result['last_measurement'] = dates_of_selected_passes[axis_slice]
        return pandas.DataFrame(result)


def get_pass_passage_time(
        selected_passes: numpy.ndarray,
        polygon: pyinterp.geodetic.Polygon | None) -> numpy.ndarray:
    """Return the passage time of the selected passes.

    Args:
        selected_passes: Selected passes.
        polygon: Polygon used to select the passes.

    Returns:
        Passage time of the selected passes.
    """
    passes = numpy.array(sorted(set(selected_passes['pass_number']))) - 1
    with xarray.open_dataset(DATASET) as ds:
        lon = ds.line_string_lon[passes, :].values
        lat = ds.line_string_lat[passes, :].values
        pass_time = ds.pass_time[passes, :].values

    result: numpy.ndarray = numpy.ndarray((len(passes), ),
                                          dtype=[('pass_number', numpy.uint16),
                                                 ('first_time', 'm8[ns]'),
                                                 ('last_time', 'm8[ns]')])

    jx = 0

    for ix, pass_index in enumerate(passes):
        line_string = pyinterp.geodetic.LineString([
            pyinterp.geodetic.Point(x, y)
            for x, y in zip(lon[ix, :], lat[ix, :])
            if numpy.isfinite(x) and numpy.isfinite(y)
        ])
        intersection = polygon.intersection(
            line_string) if polygon else line_string
        if intersection:
            lat_nadir = ds.lat_nadir[pass_index, :].values
            # Remove NaN values
            lat_nadir = lat_nadir[numpy.isfinite(lat_nadir)]
            if lat_nadir[0] > lat_nadir[-1]:
                lat_nadir = lat_nadir[::-1]
            y0 = intersection[0].lat
            y1 = intersection[1].lat if len(intersection) > 1 else y0
            t0 = numpy.searchsorted(lat_nadir, y0)
            t1 = numpy.searchsorted(lat_nadir, y1)
            selected_time = pass_time[ix, :]
            result[jx]['pass_number'] = pass_index + 1
            result[jx]['first_time'] = selected_time[min(t0, t1)]
            result[jx]['last_time'] = selected_time[max(t0, t1)]
            jx += 1

    return pandas.DataFrame(result[:jx])
