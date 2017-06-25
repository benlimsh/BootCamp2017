# solutions.py
"""Volume IB: Testing.
<Name>
<Date>
"""
import numpy as np
import itertools
import math

# Problem 1 Write unit tests for addition().
# Be sure to install pytest-cov in order to see your code coverage change.


def addition(a, b):
    return a + b


def smallest_factor(n):
    """Finds the smallest prime factor of a number.
    Assume n is a positive integer.
    """
    if n == 1:
        return 1
    for i in range(2, int(math.sqrt(n)) + 1):
        if n % i == 0:
            return i
    return n


# Problem 2 Write unit tests for operator().
def operator(a, b, oper):
    if type(oper) != str:
        raise ValueError("Oper should be a string")
    if len(oper) != 1:
        raise ValueError("Oper should be one character")
    if oper == "+":
        return a + b
    if oper == "/":
        if b == 0:
            raise ValueError("You can't divide by zero!")
        return a/float(b)
    if oper == "-":
        return a-b
    if oper == "*":
        return a*b
    else:
        raise ValueError("Oper can only be: '+', '/', '-', or '*'")

# Problem 3 Write unit test for this class.
class ComplexNumber(object):
    def __init__(self, real=0, imag=0):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def norm(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __add__(self, other):
        real = self.real + other.real
        imag = self.imag + other.imag
        return ComplexNumber(real, imag)

    def __sub__(self, other):
        real = self.real - other.real
        imag = self.imag - other.imag
        return ComplexNumber(real, imag)

    def __mul__(self, other):
        real = self.real*other.real - self.imag*other.imag
        imag = self.imag*other.real + other.imag*self.real
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        if other.real == 0 and other.imag == 0:
            raise ValueError("Cannot divide by zero")
        bottom = (other.conjugate()*other*1.).real
        top = self*other.conjugate()
        return ComplexNumber(top.real / bottom, top.imag / bottom)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __str__(self):
        return "{}{}{}i".format(self.real, '+' if self.imag >= 0 else '-',
                                                                abs(self.imag))

# Problem 5: Write code for the Set game here
#Converts strings from text file into an array of strings
#Takes in text file, returns array of strings
def file_to_array(filename):
    with open(filename,'r') as f:
        stringarray = []
        for line in f:
            line = line.split() # to deal with blank
            if line:            # lines (ie skip them)
                line = [i for i in line]
                stringarray.append(line)
    return stringarray

#Converts string representing a number into an array of its individual digits
#Takes in string, returns array
def number_to_digits(s):
    stringarray = [int(d) for d in s]
    if not (len(stringarray[i]) == 4 for i in range(0,len(stringarray))):
        raise ValueError("Cards must contain only four digits")
    return stringarray
'''
def bool_number_to_digits(s):
    stringarray = [int(d) for d in s]
    if (len(stringarray[i]) == 4 for i in range(0,len(stringarray))):
        return True
    else:
        return False
'''
#Converts each string in an array into an array of single digits, resulting
#in a larger array of digits
def stringarray_to_digitarray(stringarray):
    '''
    if bool_number_to_digits(stringarray) == False:
        raise ValueError("Cards must contain only four digits")
    '''
    digitarray =  number_to_digits(stringarray[0][0])

    for s in stringarray[1:]:
        microarray = [number_to_digits(s[0])]
        digitarray = np.vstack((digitarray,microarray))

    return digitarray

#Sums each column of a matrix
def sumcolumn(mat):
    return np.sum(mat, axis=0)

#Takes a set of three cards represented by three 4-digit number and returns whether a card is valid
#A set is a 1x3 array of 4-digit number_to_digits
def isvalid(set):
    if all(i % 3 == 0 for i in sumcolumn(set)):
        return True
    else:
        return False

#Returns the number of valid sets in a text file
def countvalid(filename):
    stringarray = file_to_array(filename)
    digitarray = stringarray_to_digitarray(stringarray) #digitarray is 12x4 matrix, need to check if array is valid

    '''
    if np.unique(digitarray).size < len(digitarray):
        raise ValueError("Cards must be unique")
    '''
    if not len(digitarray) == 12:
        raise ValueError("There must be 12 cards")

    for j in range(0,12):
        if not np.all(i ==0 or i == 1 or i == 2 for i in digitarray[j]):
            raise ValueError("Card must contain only digits 0,1,2")

    else:
        iterlist = list(itertools.combinations(digitarray, 3))

        count = 0

        for i in range(0, len(iterlist)):
            if isvalid((iterlist)[i]):
                count += 1

    return count

print(countvalid('/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards5.txt'))
