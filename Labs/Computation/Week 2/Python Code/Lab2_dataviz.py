import numpy as np
from matplotlib import pyplot as plt
from scipy import special
from matplotlib.colors import LogNorm
import matplotlib as mpl
label_size = 7
mpl.rcParams['ytick.labelsize'] = label_size

#Problem 1
def anscombe():
    ansarr = np.load('anscombe.npy')
    one_x = ansarr[:, 0]
    one_y = ansarr[:, 1]
    two_x =  ansarr[:, 2]
    two_y = ansarr[:, 3]
    three_x = ansarr[:, 4]
    three_y = ansarr[:, 5]
    four_x = ansarr[:, 6]
    four_y = ansarr[:, 7]
    x = np.linspace(0,20,100000)
    y = 1/2*x*1.0 + 3

    plt.subplot(221)
    plt.title("I")
    plt.plot(one_x, one_y, "ko", markersize = 5)
    plt.plot(x, y)

    plt.subplot(222)
    plt.title("II")
    plt.plot(two_x, two_y, "ko", markersize = 5)
    plt.plot(x, y)

    plt.subplot(223)
    plt.title("III")
    plt.plot(three_x, three_y, "ko", markersize = 5)
    plt.plot(x, y)

    plt.subplot(224)
    plt.title("IV")
    plt.plot(four_x, four_y, "ko", markersize = 5)
    plt.plot(x, y)

    plt.tight_layout()
    plt.show()

    return

anscombefindings = '''
I is mostly linear and increasing in y but has some dispersion.
II is parabolic and concave.
III is very linear with one outlier with a high y-value.
IV is vertical as there is only one x value for multiple y-values.
'''

print(anscombefindings)

#Problem 2
def getT(n,v,x):
    T = special.binom(n,v)*(x**v)*((1-x)**(n-v))
    return T

def bernstein():
    x = np.linspace(0, 1, 1000)
    for n in range(0, 4):
        for v in range(0, n+1):
            plt.subplot(4, 4, 1 + v + n*4)
            y = getT(n, v, x)
            plt.plot(x, y, lw = 2)
            plt.axis([0, 1, 0, 1])
            plt.tick_params(which = "both", top = "off", right = "off")
            if n < 2:
                plt.tick_params(labelbottom = "off")
            if n % 5:
                plt.tick_params(labelleft = "off")
            plt.title("n =" + str(n))

    plt.tight_layout()
    plt.show()

#Problem 3
# HEIGHT WEIGHT AGE
def MLB():
    mlbarr = np.load('MLB.npy')
    height = mlbarr[:,0]
    weight = mlbarr[:,1]
    age = mlbarr[:,2]
    plt.subplot(131)
    plt.plot(height, weight, 'o', markersize = 1)
    plt.plot(np.unique(height), np.poly1d(np.polyfit(height, weight, 1))(np.unique(height)))
    plt.xlabel('height')
    plt.ylabel('weight')
    plt.axis("equal")

    plt.subplot(132)
    plt.plot(height, age, 'o', markersize = 1)
    plt.plot(np.unique(height), np.poly1d(np.polyfit(height, age, 1))(np.unique(height)))
    plt.xlabel('height')
    plt.ylabel('age')
    plt.axis("equal")

    plt.subplot(133)
    plt.plot(age, weight, 'o', markersize = 1)
    plt.plot(np.unique(age), np.poly1d(np.polyfit(age, weight, 1))(np.unique(age)))
    plt.xlabel('age')
    plt.ylabel('weight')
    plt.axis("equal")
    plt.suptitle("Correlations of NBA players")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    plt.show()

    return

#DATE MAGNITUDE LONGITUDE LATITUDE
#Problem 4
def earthquakes():
    year, magnitude, longitude, latitude = np.load('earthquakes.npy').T

    plt.subplot(131)
    plt.hist(year, bins = 11, edgecolor='black', linewidth=1.2)
    plt.xlabel("Year")
    plt.ylabel("Number of Earthquakes")
    plt.title("Qn 1")

    plt.subplot(132)
    plt.hist(magnitude, bins = 10, edgecolor='black', linewidth=1.2)
    plt.title("Qn 2")

    plt.subplot(133)
    plt.plot(longitude, latitude, "o", markersize = 0.05)
    plt.xlabel('latitude')
    plt.ylabel('longitude')
    plt.axis("equal")
    plt.title("Qn 3")

    plt.suptitle("Distribution of Earthquake Data")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    return

#Problem 5
def rosenbrock():
    x = np.linspace(-10,10,1000)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    Z = (1-X)**2 + 100*((Y-X**2)**2)
    plt.contourf(X, Y, Z, 1000, cmap = "plasma", norm=LogNorm())
    plt.plot(1, 1, 'bx')
    plt.axis([-2, 2, -2, 2])
    plt.colorbar()
    plt.title("Rosenbrock Function")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

#POPN GDP MALEHEIGHT FEMALEHEIGHT
#ROWS ARE COUNTRIES
#Problem 6
def countries():
    df = np.load('countries.npy')
    population, gdp, maleheight, femaleheight = np.load('countries.npy').T
    labels = ["Austria", "Bolivia", "Brazil", "China",
            "Finland", "Germany", "Hungary", "India",
            "Japan", "North Korea", "Montenegro", "Norway",
            "Peru", "South Korea", "Sri Lanka", "Switzerland",
            "Turkey", "United Kingdom", "United States", "Vietnam"]

    plt.subplot(221)
    plt.scatter(gdp, maleheight)
    plt.plot(np.unique(gdp), np.poly1d(np.polyfit(gdp, maleheight, 1))(np.unique(gdp)))
    plt.title("Male Height vs. GDP")
    plt.subplot(222)
    plt.scatter(gdp, femaleheight)
    plt.title("Female Height vs. GDP")
    plt.plot(np.unique(gdp), np.poly1d(np.polyfit(gdp, femaleheight, 1))(np.unique(gdp)))
    plt.subplot(223)
    positions = np.arange(len(labels))
    plt.barh(positions, gdp, align = "center")
    plt.yticks(positions, labels)
    plt.title("GDP by country")
    plt.subplot(224)
    plt.hist(gdp/population, bins = 20, edgecolor = "black")
    plt.title("Distribution of GDP per capita")
    plt.tight_layout()

    plt.show()

countries()
countriesfindings = '''
There is a slight positive linear relationship between height and GDP.
Countries with higher GDP's tend to have greater average height for both males and females.
The United States has by far the highest GDP, followed by China who has about 2/3 of US GDP.
The poorest countries by GDP are Montenegro, Bolivia, and North Korea.
From the histogram, one can see that there is huge per capita income inequality between countries.
The distribution of per capita GDP is thick in the left tail and thin in the right tail.
'''
print(countriesfindings)
