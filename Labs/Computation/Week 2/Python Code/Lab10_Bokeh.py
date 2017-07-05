import numpy as np
import bokeh as bk
import pandas as pd
import pickle
from pyproj import Proj
from pyproj import transform

#Problem 2
fars = np.load('FARS.npy')
accidents = pd.read_pickle('/Users/benjaminlim/Documents/BootCamp2017/Labs/Computation/Week 2/fars_data/final_accidents2.pickle')
drivers = pd.read_pickle('/Users/benjaminlim/Documents/BootCamp2017/Labs/Computation/Week 2/fars_data/final_drivers.pickle')

from_proj = Proj(init="epsg:4326")
to_proj = Proj(init="epsg:3857")
def convert(longitudes, latitudes):
    """Converts latlon coordinates to meters.
    Inputs:
        longitudes (array-like) : array of longitudes
        latitudes (array-like) : array of latitudes
    Example:
        x,y = convert(accidents.LONGITUD, accidents.LATITUDE)
    """
    x_vals = []
    y_vals = []
    for lon, lat in zip(longitudes, latitudes):
        x, y = transform(from_proj, to_proj, lon, lat)
        x_vals.append(x)
        y_vals.append(y)
    return x_vals, y_vals
accidents["x"], accidents["y"] = convert(accidents.LONGITUD, accidents.LATITUDE)

#Problem 3
merged = drivers

#Problem 4
from bokeh.plotting import Figure
from bokeh.models import WMTSTileSource
fig = Figure(plot_width=1100, plot_height=650,
            x_range=(-13000000, -7000000), y_range=(2750000, 6250000),
            tools=["wheel_zoom", "pan"], active_scroll="wheel_zoom")
fig.axis.visible = False

STAMEN_TONER_BACKGROUND = WMTSTileSource(url='http://tile.stamen.com/toner-background/{Z}/{X}/{Y}.png', \
                                         attribution=('Map tiles by <a href="http://stamen.com">Stamen \
                                                      Design</a>, ' 'under <a href="http://creativecommons.org/licenses/by/3.0"> \
                                                      CC BY 3.0</a>.' 'Data by <a href="http://openstreetmap.org">OpenStreetMap</a>, \
                                                      ' 'under <a href="http://www.openstreetmap.org/copyright">ODbL</a>' \
                                                      )
                                         )
fig.add_tile(STAMEN_TONER_BACKGROUND)

print(accidents.describe())

#Problem 6

from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
fig = figure(plot_width=500, plot_height=500, webgl = True)
accidents_drunk = accidents[(accidents.DRUNK_DR == 1) & (accidents.SP == 0)]
accidents_speed = accidents[(accidents.DRUNK_DR == 0) & (accidents.SP == 1)]
accidents_drunkspeed = accidents[(accidents.DRUNK_DR == 1) & (accidents.SP == 1)]
cir_source_drunk = ColumnDataSource(accidents_drunk)
cir_source_speed = ColumnDataSource(accidents_speed)
cir_source_drunkspeed = ColumnDataSource(accidents_drunkspeed)
cir_drunk = fig.circle(x = "LONGITUD", y = "LATITUDE", source = cir_source_drunk, size = 0.1, fill_color = "red", fill_alpha = .1, line_alpha = .1, line_color = "red", line_width = 3)
cir_speed = fig.circle(x = "LONGITUD", y = "LATITUDE", source = cir_source_speed, size = 0.1, fill_color = "blue", fill_alpha = .1, line_alpha = .1, line_color = "blue", line_width = 3)
cir_drunkspeed = fig.circle(x = "LONGITUD", y = "LATITUDE", source = cir_source_drunkspeed, size = 0.1, fill_color = "green", fill_alpha = .1, line_alpha =.1, line_color = "green", line_width = 3)

show(fig)
