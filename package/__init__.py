from __future__ import annotations

import datetime
import pathlib

import IPython.display
import ipyleaflet
import ipywidgets
import numpy
import pandas
import pyinterp.geodetic
import xarray

from . import orf

ORF = pathlib.Path(__file__).parent / 'SWOT_ORF.txt'

ORBIT = pathlib.Path(__file__).parent / 'SWOT_orbit.nc'

PASSES_PER_CYCLE = 584

# List of HTML colors
COLORS: list[str] = [
    'aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure', 'beiae',
    'bisque', 'black', 'blanchedalmond', 'blue', 'blueviolet', 'brown',
    'burlywood', 'cadetblue', 'chartreuse', 'chocolate', 'coral',
    'cornflowerblue', 'cornsilk', 'crimson', 'cyan', 'darkblue', 'darkcyan',
    'darkgoldenrod', 'darkgray', 'darkgrey', 'darkgreen', 'darkkhaki',
    'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred',
    'darksalmon', 'darkseagreen', 'darkslateblue', 'darkslategray',
    'darkslategrey', 'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
    'dimgray', 'dimgrey', 'dodgerblue', 'firebrick', 'floralwhite',
    'forestgreen', 'fuchsia', 'gainsboro', 'ghostwhite', 'gold', 'goldenrod',
    'gray', 'grey', 'green', 'greenyellow', 'honeydew', 'hotpink',
    'indianred ', 'indigo ', 'ivory', 'khaki', 'lavender', 'lavenderblush',
    'lawngreen', 'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
    'lightgoldenrodyellow', 'lightgray', 'lightgrey', 'lightgreen',
    'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
    'lightslategray', 'lightslategrey', 'lightsteelblue', 'lightyellow',
    'lime', 'limegreen', 'linen', 'magenta', 'maroon', 'mediumaquamarine',
    'mediumblue', 'mediumorchid', 'mediumpurple', 'mediumseagreen',
    'mediumslateblue', 'mediumspringgreen', 'mediumturquoise',
    'mediumvioletred', 'midnightblue', 'mintcream', 'mistyrose', 'moccasin',
    'navajowhite', 'navy', 'oldlace', 'olive', 'olivedrab', 'orange',
    'orangered', 'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
    'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink', 'plum',
    'powderblue', 'purple', 'rebeccapurple', 'red', 'rosybrown', 'royalblue',
    'saddlebrown', 'salmon', 'sandybrown', 'seagreen', 'seashell', 'sienna',
    'silver', 'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
    'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
    'transparent', 'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
    'yellow', 'yellowgreen'
]


class DateSelection():

    def __init__(self) -> None:
        self.selection = None
        self.start_date = ipywidgets.DatePicker(description='First date:',
                                                disabled=False,
                                                value=datetime.date.today())
        self.last_date = ipywidgets.DatePicker(description='Last date:',
                                               disabled=False,
                                               value=datetime.date.today() +
                                               datetime.timedelta(days=1))

    def display(self) -> ipywidgets.Widget:
        return ipywidgets.VBox([self.start_date, self.last_date])

    def values(self) -> tuple[numpy.datetime64, numpy.timedelta64]:
        return numpy.datetime64(self.start_date.value), numpy.datetime64(
            self.last_date.value) - numpy.datetime64(self.start_date.value)


class MapSelection():

    def __init__(self) -> None:
        super().__init__()
        self.selection = None
        layout = ipywidgets.Layout(width='100%', height='600px')

        draw_control = ipyleaflet.DrawControl()
        draw_control.polyline = {}
        draw_control.polygon = {}
        draw_control.circlemarker = {}
        draw_control.rectangle = {'shapeOptions': {'color': '#0000FF'}}
        draw_control.circle = {}
        draw_control.edit = False

        self.layers: list[ipyleaflet.Polygon] = []
        self.markers: list[ipyleaflet.Marker] = []

        self.date_selection = DateSelection()
        self.search = ipywidgets.Button(description='Search')

        self.m = ipyleaflet.Map(center=[0, 0],
                                zoom=2,
                                layout=layout,
                                projection=ipyleaflet.projections.EPSG4326)
        self.m.scroll_wheel_zoom = True
        self.m.add_control(ipyleaflet.FullScreenControl())
        self.m.add_control(draw_control)
        self.m.add_control(
            ipyleaflet.WidgetControl(widget=self.date_selection.display(),
                                     position='topright'))
        self.m.add_control(
            ipyleaflet.WidgetControl(widget=self.search,
                                     position='bottomright'))
        self.out = ipywidgets.Output()

        self.main_widget = ipywidgets.VBox([self.m, self.out])
        draw_control.on_draw(self.handle_draw)
        self.search.on_click(self.handle_compute)
        self.widget_error = None

    def display(self) -> ipywidgets.Widget:
        return self.main_widget

    def handle_error(self, *args) -> None:
        self.m.remove_control(self.m.controls[-1])
        self.widget_error = None

    def clear_last_selection(self) -> None:
        for item in self.markers:
            self.m.remove(item)
        self.markers.clear()
        for item in self.layers:
            self.m.remove(item)
        self.layers.clear()
        self.selection = None

    def handle_draw(self, target, action, geo_json) -> None:
        if action == 'deleted':
            self.clear_last_selection()
            return

        if action != 'created':
            return

        self.clear_last_selection()

        try:
            coordinates = geo_json['geometry']['coordinates']

            # Build a polygon with interpolated longitudes between the first and
            # last points to restrict the search area to the latitude of the
            # selected zone.
            x = [item[0] for item in coordinates[0]]
            y = [item[1] for item in coordinates[0]]
            x0, x1 = x[0], x[2]
            y0, y1 = y[0], y[1]
            xs = numpy.linspace(x0, x1, round(x1 - x0) * 2, endpoint=True)
            points = [
                pyinterp.geodetic.Point(item, y0) for item in reversed(xs)
            ] + [pyinterp.geodetic.Point(item, y1) for item in xs]
            points.append(points[0])
            self.selection = pyinterp.geodetic.Polygon(points)
        except (KeyError, IndexError):
            self.selection = None

    def display_error(self, msg) -> None:
        panel = ipywidgets.HTML(msg)
        close = ipywidgets.Button(description='Close.',
                                  disabled=False,
                                  button_style='danger')
        self.widget_error = ipywidgets.VBox([panel, close])
        assert self.widget_error is not None
        self.widget_error.box_style = 'danger'
        self.widget_error.layout = ipywidgets.Layout(display='flex',
                                                     flex_flow='column',
                                                     align_items='center',
                                                     border='5')
        close.on_click(self.handle_error)
        self.m.add_control(
            ipyleaflet.WidgetControl(widget=self.widget_error,
                                     position='bottomright'))

    def handle_compute(self, args) -> None:
        try:
            if self.selection is None:
                if self.widget_error is None:
                    self.display_error(
                        'Please select an area by drawing a rectangle on the '
                        'map, then click on the <b>Compute</b> button.')
                self.search.on_click(self.handle_compute)

                return
            self.out.clear_output()
            with self.out:
                IPython.display.display('Computing...')
            selected_passes = compute_selected_passes(self.date_selection,
                                                      self)

            self.markers, self.layers = plot_selected_passes(
                self, selected_passes)
            for item in self.layers:
                self.m.add_layer(item)
            for item in self.markers:
                self.m.add_layer(item)

            self.out.clear_output()
            with self.out:
                IPython.display.display(selected_passes)
        except Exception as err:
            import traceback
            self.display_error(str(err) + '<br>' + traceback.format_exc())


class MainWidget():

    def __init__(self) -> None:
        self.map_selection = MapSelection()
        self.date_selection = DateSelection()
        self.button = ipywidgets.Button(description='Compute')
        self.panel = ipywidgets.VBox([
            ipywidgets.HBox(
                [self.map_selection.display(),
                 self.date_selection.display()]), self.button
        ])

    def display(self) -> ipywidgets.Widget:
        return self.panel


def get_cycle_duration(dataset: xarray.Dataset) -> numpy.timedelta64:
    start_time = dataset.start_time[0].values
    end_time = dataset.end_time[-1].values
    return end_time - start_time


def calculate_cycle_axis(
        cycle_duration: numpy.timedelta64) -> pyinterp.TemporalAxis:
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
    with xarray.open_dataset(ORBIT.resolve()) as ds:
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
    passes = numpy.array(sorted(set(selected_passes['pass_number']))) - 1
    with xarray.open_dataset(ORBIT) as ds:
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


def _load_one_polygons(x, y):
    m = numpy.isfinite(x) & numpy.isfinite(y)
    x = x[m]
    y = y[m]
    return pyinterp.geodetic.Polygon(
        [pyinterp.geodetic.Point(x, y) for x, y in zip(x, y)])


def load_polygons(pass_number: numpy.ndarray):
    index = pass_number - 1

    left_polygon = []
    right_polygon = []

    with xarray.open_dataset(ORBIT) as ds:
        for ix in index:
            left_polygon.append(
                (ix + 1,
                 _load_one_polygons(ds.left_polygon_lon[ix, :].values,
                                    ds.left_polygon_lat[ix, :].values)))
            right_polygon.append(
                (ix + 1,
                 _load_one_polygons(ds.right_polygon_lon[ix, :].values,
                                    ds.right_polygon_lat[ix, :].values)))
    return left_polygon, right_polygon


def compute_selected_passes(date_selection: DateSelection,
                            map_selection: MapSelection):
    if map_selection.selection is None:
        raise ValueError('No area selected.')
    first_date, search_duration = date_selection.values()
    if search_duration < numpy.timedelta64(0, 'D'):
        raise ValueError('First date must be before last date.')
    selected_passes = get_selected_passes(first_date, search_duration)
    pass_passage_time = get_pass_passage_time(selected_passes,
                                              map_selection.selection)
    selected_passes = selected_passes.join(
        pass_passage_time.set_index('pass_number'),
        on='pass_number',
        how='right')
    selected_passes.sort_values(by=['cycle_number', 'pass_number'],
                                inplace=True)
    selected_passes['first_measurement'] += selected_passes['first_time']
    selected_passes['last_measurement'] += selected_passes['last_time']
    selected_passes.drop(columns=['first_time', 'last_time'], inplace=True)
    selected_passes['first_measurement'] = selected_passes[
        'first_measurement'].dt.floor('s')
    selected_passes['last_measurement'] = selected_passes[
        'last_measurement'].dt.floor('s')
    selected_passes.reset_index(drop=True, inplace=True)
    return selected_passes


def plot_selected_passes(map_selection: MapSelection,
                         df: pandas.DataFrame) -> tuple[list, list]:
    polygon = map_selection.selection
    bbox = (polygon if polygon is not None else
            pyinterp.geodetic.Box.whole_earth().as_polygon())

    left_swath, right_swath = load_polygons(df['pass_number'].values)

    layers = []
    markers = []

    for pass_number, item in left_swath:
        item = item.intersection(bbox)
        if len(item) == 0:
            continue
        outer = item[0].outer
        color_id = pass_number % len(COLORS)
        poly = ipyleaflet.Polygon(
            locations=[(p.lat, p.lon) for p in outer],
            color=COLORS[color_id],
            fill_color=COLORS[color_id],
        )
        marker = ipyleaflet.Marker(location=(outer[1].lat, outer[1].lon))
        marker.popup = ipywidgets.HTML(f'Pass {pass_number}')

        markers.append(marker)
        layers.append(poly)

    for pass_number, item in right_swath:
        item = item.intersection(bbox)
        if len(item) == 0:
            continue
        outer = item[0].outer
        color_id = pass_number % len(COLORS)
        poly = ipyleaflet.Polygon(
            locations=[(p.lat, p.lon) for p in outer],
            color=COLORS[color_id],
            fill_color=COLORS[color_id],
        )
        layers.append(poly)

    return markers, layers
