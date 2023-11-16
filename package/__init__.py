from __future__ import annotations

import base64
import dataclasses
import datetime
import traceback

import IPython.display
import ipyleaflet
import ipywidgets
import numpy
import pandas
import pyinterp.geodetic
import xarray

from . import orbit

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

#: HTML Template for the popup of the marker
POPUP_TEMPLATE = """
<div style="text-align: center; font-weight: bold;">
    <div style="display: inline-block; width: 10px; height: 10px;
    border: 1px solid black; background-color: {color};
    margin-right: 5px;"></div>
    Pass {pass_number}
</div>
"""

#: HTML Template for the popup of the marker
DOWNLOAD_TEMPLATE = """<a href="data:file/csv;base64,{b64}"
download="selected_passes.csv"><button style="background-color: #4285F4;
color: white; border-radius: 4px; padding: 10px 16px; font-size: 14px;
font-weight: bold; border: none; cursor: pointer;">
Download data as a CSV file</button></a>"""

#: Type of a pass polygon
PassPolygon = tuple[int, pyinterp.geodetic.Polygon]


@dataclasses.dataclass
class Swath:
    left: ipyleaflet.Polygon
    right: ipyleaflet.Polygon
    marker: ipyleaflet.Marker


class InvalidDate(Exception):
    pass


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
        self.bounds = [[-180, -90], [180, 90]]
        layout = ipywidgets.Layout(width='100%', height='600px')

        draw_control = ipyleaflet.DrawControl()
        draw_control.polyline = {}
        draw_control.polygon = {}
        draw_control.circlemarker = {}
        draw_control.rectangle = {'shapeOptions': {'color': '#0000FF'}}
        draw_control.circle = {}
        draw_control.edit = False

        self.swaths: list[Swath] = []

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
        self.widget_error: ipywidgets.VBox | None = None

    def display(self) -> ipywidgets.Widget:
        return self.main_widget

    def handle_error(self, *args) -> None:
        self.m.remove_control(self.m.controls[-1])
        self.widget_error = None
        self.search.disabled = False

    def clear_last_layers(self) -> None:
        for item in self.swaths:
            self.m.remove(item.left)
            self.m.remove(item.right)
            self.m.remove(item.marker)
        self.swaths.clear()
        self.out.clear_output()

    def clear_last_selection(self) -> None:
        self.clear_last_layers()
        self.selection = None
        self.bounds = [[-180, -90], [180, 90]]

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
            x = numpy.array([item[0] for item in coordinates[0]])
            y = numpy.array([item[1] for item in coordinates[0]])
            x0, x1 = x[0], x[2]
            y0, y1 = y[0], y[1]
            xs = numpy.linspace(x0, x1, round(x1 - x0) * 2, endpoint=True)
            self.bounds = [[min(x), min(y)], [max(x), max(y)]]
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
        self.search.disabled = True

    def handle_compute(self, args) -> None:
        try:
            if self.selection is None:
                if self.widget_error is None:
                    self.display_error(
                        'Please select an area by drawing a rectangle on the '
                        'map, then click on the <b>Search</b> button.')
                return
            self.clear_last_layers()
            with self.out:
                IPython.display.display('Computing...')
            selected_passes = compute_selected_passes(self.date_selection,
                                                      self)

            self.swaths = plot_selected_passes(self, selected_passes)
            # Rename the columns "first_measurement" and "last_measurement"
            # to "first date" and "last date"
            selected_passes.rename(
                columns={
                    'first_measurement': 'First date',
                    'last_measurement': 'Fast date',
                    'cycle_number': 'Cycle number',
                    'pass_number': 'Pass number'
                },
                inplace=True,
            )

            for item in self.swaths:
                self.m.add_layer(item.left)
                self.m.add_layer(item.right)
                self.m.add_layer(item.marker)
            self.out.clear_output()
            with self.out:
                IPython.display.display(selected_passes)
                # Generate a link to download the data as a CSV file.
                csv = selected_passes.to_csv(sep=';', index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                IPython.display.display(
                    ipywidgets.HTML(DOWNLOAD_TEMPLATE.format(b64=b64)))
        except InvalidDate as err:
            self.out.clear_output()
            self.display_error(str(err))
        except Exception as err:
            self.out.clear_output()
            self.display_error(
                str(err) + '<br>'.join(traceback.format_exc().splitlines()))


class MainWidget:

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


def _load_one_polygons(x, y) -> pyinterp.geodetic.Polygon:
    m = numpy.isfinite(x) & numpy.isfinite(y)
    x = x[m]
    y = y[m]
    return pyinterp.geodetic.Polygon(
        [pyinterp.geodetic.Point(x, y) for x, y in zip(x, y)])


def load_polygons(
        pass_number: numpy.ndarray
) -> tuple[list[PassPolygon], list[PassPolygon]]:
    index = pass_number - 1

    left_polygon: list[PassPolygon] = []
    right_polygon: list[PassPolygon] = []

    with xarray.open_dataset(orbit.DATASET) as ds:
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
                            map_selection: MapSelection) -> pandas.DataFrame:
    if map_selection.selection is None:
        raise ValueError('No area selected.')
    first_date, search_duration = date_selection.values()
    if search_duration < numpy.timedelta64(0, 'D'):
        raise InvalidDate('First date must be before last date.')
    selected_passes = orbit.get_selected_passes(first_date, search_duration)
    pass_passage_time = orbit.get_pass_passage_time(selected_passes,
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


def plot_swath(
    pass_number: int,
    item: pyinterp.geodetic.Polygon,
    bbox: pyinterp.geodetic.Polygon,
    layers: dict[int, ipyleaflet.Polygon],
    markers: dict[int, ipyleaflet.Marker],
    east: float,
) -> None:
    intersection = item.intersection(bbox)
    if len(intersection) == 0:
        return
    outer = intersection[0].outer

    lons = numpy.array([p.lon for p in outer])
    lats = numpy.array([p.lat for p in outer])
    lons = numpy.deg2rad(
        pyinterp.geodetic.normalize_longitudes(
            numpy.array([p.lon for p in outer]), east))
    lons = numpy.unwrap(lons, discont=numpy.pi)
    lons = numpy.rad2deg(lons)

    color_id = pass_number % len(COLORS)
    layers[pass_number] = ipyleaflet.Polygon(
        locations=[(y, x) for x, y in zip(lons, lats)],
        color=COLORS[color_id],
        fill_color=COLORS[color_id],
    )
    if pass_number not in markers:
        size = lons.size // 8
        index = max(size, 0) if pass_number % 2 == 0 else min(
            size * 7, size - 1)
        marker = ipyleaflet.Marker(location=(lats[index], lons[index]))
        marker.draggable = False
        marker.opacity = 0.8
        marker.popup = ipywidgets.HTML(
            POPUP_TEMPLATE.format(color=COLORS[color_id],
                                  pass_number=pass_number))
        markers[pass_number] = marker


def plot_selected_passes(map_selection: MapSelection,
                         df: pandas.DataFrame) -> list[Swath]:
    polygon = map_selection.selection
    bbox: pyinterp.geodetic.Polygon = (  # type: ignore[assignment]
        polygon if polygon is not None else
        pyinterp.geodetic.Box.whole_earth().as_polygon())

    (left_swath, right_swath) = load_polygons(
        df['pass_number'].values)  # type: ignore[arg-type]

    left_layers: dict[int, ipyleaflet.Polygon] = {}
    right_layers: dict[int, ipyleaflet.Polygon] = {}
    markers: dict[int, ipyleaflet.Marker] = {}
    east = map_selection.bounds[0][0]
    for pass_number, item in left_swath:
        plot_swath(pass_number, item, bbox, left_layers, markers, east)

    for pass_number, item in right_swath:
        plot_swath(pass_number, item, bbox, right_layers, markers, east)

    layers: list[Swath] = [
        Swath(left=left_layers.get(pass_number, ipyleaflet.Polygon()),
              right=right_layers.get(pass_number, ipyleaflet.Polygon()),
              marker=marker) for pass_number, marker in markers.items()
    ]
    return layers
