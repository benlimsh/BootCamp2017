import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.close("all")

#Problem 1
def evenseries():
    even = np.array([2,4,6,8,10])
    s1 = pd.Series(-3, index = even)

    d = {'Bill': 31, 'Sarah': 28, 'Jane': 34, 'Joe': 26}
    s2 = pd.Series(d)

    return [s1, s2]


#Problem 2
def randomwalk():
    N = 100
    plt.subplot(141)
    s1 = np.zeros(N)
    s1[1:] = np.random.binomial(1, .5, size = (N-1,))*2-1
    s1 = pd.Series(s1)
    s1 = s1.cumsum()
    plt.plot(s1)

    s2 = np.zeros(N)
    s2[1:] = np.random.binomial(1, .5, size = (N-1,))*2-1
    s2 = pd.Series(s2)
    s2 = s2.cumsum()
    plt.plot(s2)

    s3 = np.zeros(N)
    s3[1:] = np.random.binomial(1, .5, size = (N-1,))*2-1
    s3 = pd.Series(s3)
    s3 = s3.cumsum()
    plt.plot(s3)

    s4 = np.zeros(N)
    s4[1:] = np.random.binomial(1, .5, size = (N-1,))*2-1
    s4 = pd.Series(s4)
    s4 = s4.cumsum()
    plt.plot(s4)

    s5 = np.zeros(N)
    s5[1:] = np.random.binomial(1, .5, size = (N-1,))*2-1
    s5 = pd.Series(s5)
    s5 = s5.cumsum()
    plt.plot(s5)
    plt.title("p=0.5,N=100")
    plt.ylim([-50,50])

    plt.subplot(142)
    N1 = 100
    s6 = np.zeros(N1)
    s6[1:] = np.random.binomial(1, .51, size = (N1-1,))*2-1
    s6 = pd.Series(s6)
    s6 = s6.cumsum()
    plt.title("p=0.51,N=100")
    plt.plot(s6)

    plt.subplot(143)
    N2 = 10000
    s7 = np.zeros(N2)
    s7[1:] = np.random.binomial(1, .51, size = (N2-1,))*2-1
    s7 = pd.Series(s7)
    s7 = s7.cumsum()
    plt.title("p=0.51,N=10000")
    plt.plot(s7)

    plt.subplot(144)
    N3 = 100000
    s8 = np.zeros(N3)
    s8[1:] = np.random.binomial(1, .51, size = (N3-1,))*2-1
    s8 = pd.Series(s8)
    s8 = s8.cumsum()
    plt.title("p=0.51,N=100000")
    plt.plot(s8)
    plt.suptitle("Comparison of Random Walks")
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()
    return

#Problem 3
name = ['Mylan', 'Regan', 'Justin', 'Jess', 'Jason', 'Remi', 'Matt', 'Alexander', 'JeanMarie']
sex = ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'M', 'F']
age = [20, 21, 18, 22, 19, 20, 20, 19, 20]
rank = ['Sp', 'Se', 'Fr', 'Se', 'Sp', 'J', 'J', 'J', 'Se']
ID = range(9)
aid = ['y', 'n', 'n', 'y', 'n', 'n', 'n', 'y', 'n']
GPA = [3.8, 3.5, 3.0, 3.9, 2.8, 2.9, 3.8, 3.4, 3.7]
mathID = [0, 1, 5, 6, 3]
mathGd = [4.0, 3.0, 3.5, 3.0, 4.0]
major = ['y', 'n', 'y', 'n', 'n']
studentInfo = pd.DataFrame({'ID': ID, 'Name': name, 'Sex': sex, 'Age': age, 'Class': rank})
otherInfo = pd.DataFrame({'ID': ID, 'GPA': GPA, 'Financial_Aid': aid})
mathInfo = pd.DataFrame({'ID': mathID, 'Grade': mathGd, 'Math_Major': major})

def query():
    age = studentInfo['Age'] > 19
    sex = studentInfo['Sex'] == 'M'
    return studentInfo[age & sex][['ID', 'Name']]

#Problem 4
def maledata():
    merged = pd.merge(studentInfo,otherInfo, on = 'ID', how = 'outer')
    return merged[merged['Sex']=='M'][['Age', 'GPA']]

#Problem 5
#skip first row
#Year,Population,Total,Violent,Property,Murder,Forcible-Rape,Robbery,Aggravated-assault,Burglary,Larcency-Theft,Vehicle-Theft

def crime():
    data = pd.read_csv('crime_data.txt', skiprows=[0], index_col = ['Year'])
    data['Rate'] = pd.Series(data['Total']/data['Population'], index = data.index)
    plt.xticks(data['Rate'], data.index.values)
    plt.plot(data['Rate'])
    plt.ylabel("Crime Rate")
    plt.xlabel("Year")
    plt.title("Crime Rate vs. Year")
    plt.show()
    plt.close()

    datacopy = data.nlargest(5, 'Rate')
    print("5 years with highest crime rate: ", datacopy.index.values)
    print(data)

    avgtotal = round(np.mean(data['Total']),1)
    avgburglary = round(np.mean(data['Burglary']),1)
    print("Average Total: ", avgtotal, "Average Burglary: ", avgburglary)

    tot = data['Total'] < avgtotal
    burg = data['Burglary'] > avgburglary
    subset = data[tot & burg]
    print("The years where the total number of crimes were below average but number of burglaries were above average are:", subset.index.values)

    plt.plot(data['Population'], data['Total'])
    plt.xlabel("Population in 1e7")
    plt.ylabel("Number of Crimes in 1e7")
    plt.title("Total Number of Crimes vs. Population Size")
    plt.show()
    plt.close()

    saveddata = data[(data.index.values > 1979) & (data.index.values < 1990)][['Population', 'Violent', 'Robbery']]
    saveddata.to_csv('crime_subset.txt')

    return

crime()
