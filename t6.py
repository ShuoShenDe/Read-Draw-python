import re
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
from descartes import PolygonPatch

f = open("Data_pro/t6/CH2017BST.txt", "r", encoding="utf8")

lines = []
x = []
y = []
name = []
for row in f:
    if row.startswith("66666"):
        name.append(row.split()[7])
        lines.append([x, y])
        x = []
        y = []
    else:
        y.append(float(row.split()[2]) / 10)
        x.append(float(row.split()[3]) / 10)

shape = gpd.read_file('ne_110m_land.shp')
shape.plot()

for line in lines:
    plt.plot(line[0], line[1], linewidth=1)

plt.ylim([10, 50])
plt.xlim([100, 180])
plt.gca().legend(name, loc='center left', bbox_to_anchor=(1.01, 0.5), fontsize=5)
plt.tight_layout()
plt.title("Typhoon Trajectory")
plt.show()
