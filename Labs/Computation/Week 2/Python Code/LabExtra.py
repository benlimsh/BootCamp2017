import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
from itertools import cycle
from array import *

def histogram():
    lipids = pd.read_csv('lipids.csv', skiprows = 4, index_col = 0)
    diseased = lipids[lipids.index==1]
    n, bin_cuts, patches = plt.hist(diseased['chol'], 25)
    plt.title("2D Histogram of Cholesterol")
    plt.xlabel('Cholesterol level')
    plt.ylabel('Frequency')
    plt.show()
    highfreq_mid = (bin_cuts[n.argmax()] + bin_cuts[n.argmax()-1])/2
    print(highfreq_mid)

    from mpl_toolkits.mplot3d import Axes3D

    '''
    --------------------------------------------------------------------
    bin_num  = integer > 2, number of bins along each axis
    hist     = (bin_num, bin_num) matrix, bin percentages
    xedges   = (bin_num+1,) vector, bin edge values in x-dimension
    yedges   = (bin_num+1,) vector, bin edge values in y-dimension
    x_midp   = (bin_num,) vector, midpoints of bins in x-dimension
    y_midp   = (bin_num,) vector, midpoints of bins in y-dimension
    elements = integer, total number of 3D histogram bins
    xpos     = (bin_num * bin_num) vector, x-coordinates of each bin
    ypos     = (bin_num * bin_num) vector, y-coordinates of each bin
    zpos     = (bin_num * bin_num) vector, zeros or z-coordinates of
                origin of each bin
    dx       = (bin_num,) vector, x-width of each bin
    dy       = (bin_num,) vector, y-width of each bin
    dz       = (bin_num * bin_num) vector, height of each bin
    --------------------------------------------------------------------
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection ='3d')
    hist, xedges, yedges = np.histogram2d(diseased['chol'], diseased['trig'], bins=25)

    x_midp = xedges[:-1] + 0.5 * (xedges[1] - xedges[0])
    y_midp = yedges[:-1] + 0.5 * (yedges[1] - yedges[0])
    elements = (len(xedges) - 1) * (len(yedges) - 1)
    ypos, xpos = np.meshgrid(y_midp, x_midp)
    xpos = xpos.flatten()
    ypos = ypos.flatten()
    zpos = np.zeros(elements)
    dx = (xedges[1] - xedges[0]) * np.ones_like(25)
    dy = (yedges[1] - yedges[0]) * np.ones_like(25)
    dz = hist.flatten()
    ax.bar3d(xpos, ypos, zpos, dx, dy, dz, color='y', zsort='average')
    ax.set_xlabel('Cholesterol Frequency')
    ax.set_ylabel('Triglyceride Frequency')
    ax.set_zlabel('Percentage')
    plt.title('3D Histogram of Cholesterol and Trigliceride')

    plt.tight_layout()
    plt.show()

    print("2b) Individuals with higher levels of cholesterol also tend to have \
          higher levels of triglyceride.")

    print("2c) Individuals with high cholesterol and high triglyceride have \
          the highest risk of heart disease.")
    return

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime

def nber():
    peaks = ["1929-08-01","1937-05-01","1945-02-01","1948-11-01","1953-07-01","1957-08-01","1960-04-01","1969-12-01","1973-11-01","1980-01-01","1981-07-01","1990-07-01","2001-03-01","2007-12-01"]
    payems = pd.read_csv('/Users/benjaminlim/Documents/BootCamp2017/Labs/Computation/Week 2/payems.csv', skiprows =[0,1,2,3,4], index_col=0)
    payems.index = pd.to_datetime(payems.index)
    peaks = pd.to_datetime(pd.Series(peaks))

    firstpeak = pd.Series([np.nan])
    start = peaks[0].replace(year=peaks[0].year-1)
    end = peaks[0].replace(year=peaks[0].year+7)
    payemscopy = payems[(payems.index >= start) & (payems.index <= end)]
    series = payemscopy['payems'].resample("12M").mean()
    series = series/series[0]
    first = firstpeak.append(series)
    first.index = range(1, 10)
    array =[]
    array.append(first)

    for i in range(1,14):
            start = peaks[i].replace(year=peaks[i].year-1)
            end = peaks[i].replace(year=peaks[i].year+7)
            payemscopy = payems[(payems.index >= start) & (payems.index <= end)]
            series = payemscopy['payems'].resample("12M").mean()
            series = series/series[1]
            series.index = range(0,len(series))
            array.append(series)

    from itertools import cycle
    lines = ["*","-","+","=",":","-."]
    linecycler = cycle(lines)

    for i in range(1,14):
        plt.plot(array[i], label = str(peaks[i].strftime("%m-%Y")))

    plt.plot(array[0],"k", lw=2, label =(str(peaks[0].strftime("%m-%Y")+ " Great Depression")))
    plt.plot(array[13],"r", lw=2, label =(str(peaks[13].strftime("%m-%Y")+ " Great Recession")))
    plt.axhline(1,linestyle = 'dashed', color = 'grey')
    plt.axvline(2,linestyle = 'dashed', color = 'grey')
    xticklabs = ['-1yr', 'peak', '+1yr', '+2yr','+3yr','+4yr','+5yr','+6yr','+7yr']
    plt.xticks(range(1,10),xticklabs)

    plt.xlabel("Time from peak")
    plt.ylabel("Jobs/peak")
    plt.title("Job growth versus Time")
    plt.legend(bbox_to_anchor=(1.4,1), loc="upper right", borderaxespad =0.)

    plt.tight_layout()
    plt.show()
    return
nber()

def ricardo():
    chicago = pd.read_csv('chicago.csv')
    dc = pd.read_csv('dc.csv')
    pitts = pd.read_csv('pittsburgh.csv')
    miami = pd.read_csv('miami.csv')
    indian = pd.read_csv('indianapolis.csv')

    chicago['STATION'] = "Chicago"
    chicago['DATE'] = pd.to_datetime(chicago['DATE'], format='%Y%m%d')
    chicago.index = chicago['DATE']
    chicago['DOY'] = chicago.index.dayofyear
    chicago['DOY']=(chicago["DOY"]-264)%366

    mask1 = (chicago['DOY'] >= 264) & (chicago['DOY'] <=365)
    mask2 = chicago['DOY'] < 264


    dc['STATION'] = "DC"
    dc['DATE'] = pd.to_datetime(dc['DATE'], format='%Y%m%d')
    dc.index = dc['DATE']
    dc['DOY'] = dc.index.dayofyear
    dc['DOY']=(dc["DOY"]-264)%366

    mask1 = (dc['DOY'] >= 264) & (dc['DOY'] <= 365)
    mask2 = dc['DOY'] < 264



    pitts['STATION'] = "Pittsburgh"
    pitts['DATE'] = pd.to_datetime(pitts['DATE'], format='%Y%m%d')
    pitts.index = pitts['DATE']
    pitts['DOY'] = pitts.index.dayofyear
    pitts['DOY']=(pitts["DOY"]-264)%366

    mask1 = (pitts['DOY'] >= 264) & (pitts['DOY'] <=365)
    mask2 = pitts['DOY'] < 264


    miami['STATION'] = "Miami"
    miami['DATE'] = pd.to_datetime(miami['DATE'], format='%Y%m%d')
    miami.index = miami['DATE']
    miami['DOY'] = miami.index.dayofyear
    miami['DOY']=(miami["DOY"]-264)%366

    mask1 = (miami['DOY'] >= 264) & (miami['DOY'] <=365)
    mask2 = miami['DOY'] < 264


    indian['STATION'] = "Indianapolis"
    indian['DATE'] = pd.to_datetime(indian['DATE'], format='%Y%m%d')
    indian.index = indian['DATE']
    indian['DOY'] = indian.index.dayofyear
    indian['DOY']=(indian["DOY"]-264)%366

    mask1 = (indian['DOY'] >= 264) & (indian['DOY'] <=365)
    mask2 = indian['DOY'] < 264


    plt.rcParams["figure.figsize"] = [11,6]

    plt.scatter(chicago['DOY'],chicago['TMAX'], s=0.2, c='maroon', alpha=0.5)
    plt.scatter(chicago['DOY'],chicago['TMIN'], s=0.2, c='maroon', alpha=0.5)
    plt.scatter(dc['DOY'], dc['TMAX'],  s=0.2, c='k', alpha=0.5)
    plt.scatter(dc['DOY'], dc['TMIN'],s=0.2, c='k', alpha=0.5)
    plt.scatter(pitts['DOY'], pitts['TMAX'], s=0.2, c='k', alpha=0.5)
    plt.scatter(pitts['DOY'], pitts['TMIN'], s=0.2, c='k', alpha=0.5)
    plt.scatter(miami['DOY'], miami['TMAX'], s=0.2, c='k', alpha=0.5)
    plt.scatter(miami['DOY'], miami['TMIN'],s=0.2, c='k', alpha=0.5)
    plt.scatter(indian['DOY'], indian['TMAX'], s=0.2, c='k', alpha=0.5)
    plt.scatter(indian['DOY'], indian['TMIN'], s=0.2, c='k', alpha=0.5)
    plt.scatter(indian['DOY']['DATE' == "19752201"], indian['TMAX']['DATE' == "19752201"], c='y' , edgecolor ='k')
    plt.scatter(indian['DOY']['DATE' == "19752201"], indian['TMIN']['DATE' == "19752201"], c='y' , edgecolor ='k')
    plt.scatter(pitts['DOY']['DATE' == "19880714"], pitts['TMAX']['DATE' == "19880714"], c='y' , edgecolor ='k')
    plt.scatter(pitts['DOY']['DATE' == "19880714"], pitts['TMIN']['DATE' == "19880714"], c='y' , edgecolor ='k')

    plt.ylim(ymin = -25, ymax = 120)
    plt.annotate('Born', xy =(indian['DOY']['DATE' == "19752201"], indian['TMAX']['DATE' == "19752201"]), xytext=(-15, 20), arrowprops=dict(facecolor='black', shrink=0.1))
    plt.annotate('little league all-star team wins regional championship', xy =(pitts['DOY']['DATE' == "19880714"], pitts['TMAX']['DATE' == "19880714"]), xytext=(100, 100), arrowprops=dict(facecolor='black', shrink=0.1))
    plt.title("Ricardo's lifetime temperature range")
    plt.ylabel("Temperatures in Fahrenheit")
    plt.xlabel("Days of the Year")
    plt.show()

    return
