from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cbook
from matplotlib import cm
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
import numpy as np
import re
import math
import scipy.ndimage.filters as filters
from scipy.signal import argrelextrema
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.feature import peak_local_max
from matplotlib.cbook import get_sample_data

#####读取文件并处理成numpy并返回
def readfile():
    ascii_grid = np.loadtxt('Data_pro/t5/dem.asc', skiprows=6)

    return ascii_grid

#####在原栅格图像周围加一圈并返回
def AddRound(npgrid):
    ny, nx = npgrid.shape  # ny:行数，nx:列数

    zbc = np.zeros((ny + 2, nx + 2))
    zbc[1:-1, 1:-1] = npgrid
    # 四边
    zbc[0, 1:-1] = npgrid[0, :]
    zbc[-1, 1:-1] = npgrid[-1, :]
    zbc[1:-1, 0] = npgrid[:, 0]
    zbc[1:-1, -1] = npgrid[:, -1]
    # 角点
    zbc[0, 0] = npgrid[0, 0]
    zbc[0, -1] = npgrid[0, -1]
    zbc[-1, 0] = npgrid[-1, 0]
    zbc[-1, -1] = npgrid[-1, 0]
    return zbc

#####计算xy方向的梯度
def Cacdxdy(npgrid, sizex, sizey):
    zbc = AddRound(npgrid)
    dx = ((zbc[1:-1, :-2]) - (zbc[1:-1, 2:])) / sizex / 2
    dy = ((zbc[2:, 1:-1]) - (zbc[:-2, 1:-1])) / sizey / 2
    dx = dx[1:-1, 1:-1]
    dy = dy[1:-1, 1:-1]
    # np.savetxt("dxdy.csv", dx, delimiter=",")
    return dx, dy

####计算坡度\坡向
def CacSlopAsp(dx, dy):
    slope = (np.arctan(np.sqrt(dx * dx + dy * dy))) * 57.29578  # 转换成°
    slope = slope[1:-1, 1:-1]
    # 坡向
    a = np.zeros([dx.shape[0], dx.shape[1]]).astype(np.float32)
    for i in range(dx.shape[0]):
        for j in range(dx.shape[1]):
            x = float(dx[i, j])
            y = float(dy[i, j])
            if (x == 0.) & (y == 0.):
                a[i, j] = -1
            elif x == 0.:
                if y > 0.:
                    a[i, j] = 0.
                else:
                    a[i, j] = 180.
            elif y == 0.:
                if x > 0:
                    a[i, j] = 90.
                else:
                    a[i, j] = 270.
            else:
                a[i, j] = float(math.atan(y / x)) * 57.29578
                if a[i, j] < 0.:
                    a[i, j] = 90. - a[i, j]
                elif a[i, j] > 90.:
                    a[i, j] = 450. - a[i, j]
                else:
                    a[i, j] = 90. - a[i, j]
    return slope, a


####绘制平面栅格图
def Drawgrid(A=[], strs="", X=[], Y=[]):
    if strs == "":
        plt.imshow(A, interpolation='nearest', cmap=plt.cm.hot, origin='lower')  # cmap='bone'  cmap=plt.cm.hot
        plt.colorbar(shrink=0.8)
    else:
        plt.imshow(A, interpolation='nearest', cmap=strs, origin='lower')  # cmap='bone'  cmap=plt.cm.hot
        plt.colorbar(shrink=0.8)
    image_path = './Data_pro/t5/flag-a.png'
    #fig, ax = plt.subplots()
    imscatter(X, Y, image_path, zoom=0.02)
    plt.xticks(())
    plt.yticks(())
    plt.show()

def plot_images(x, y, image, ax=None):
    ax = ax or plt.gca()
    for xi, yi in zip(x, y):
        im = OffsetImage(image, zoom=72 / ax.figure.dpi)
        im.image.axes = ax
        ab = AnnotationBbox(im, (xi, yi), frameon=False, pad=0.0, )
        ax.add_artist(ab)

def imscatter(x, y, image, ax=None, zoom=1):
    if ax is None:
        ax = plt.gca()
    try:
        image = plt.imread(image)
    except TypeError:
        # Likely already an array...
        pass
    im = OffsetImage(image, zoom=zoom)
    x, y = np.atleast_1d(x, y)
    artists = []
    for x0, y0 in zip(x, y):
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    return artists

####程序入口
if __name__ == '__main__':
    npgrid = readfile()
    # 得到局部最大值的坐标
    coordinates = peak_local_max(npgrid, min_distance=10)

    # get latitude and longitude
    Y = [i[0] for i in coordinates]
    X = [i[1] for i in coordinates]

    npgrid = AddRound(npgrid)
    dx, dy = Cacdxdy(npgrid, 22.5, 22.5)
    slope, arf = CacSlopAsp(dx, dy)

    # 绘制坡向图
    Drawgrid(A=arf, strs="gray", X=X, Y=Y)

