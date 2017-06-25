import numpy as np
import itertools

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
    stringarray = [int(d) for d in s] #might have to do str(s) instead
    return stringarray

#Converts each string in an array into an array of single digits, resulting
#in a larger array of digits
def stringarray_to_digitarray(stringarray):
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

#Returns if a card has digits other than 0,1,2
#Returns if a card has anything but 4 digits

#Returns the number of valid sets in a text file
def countvalid(filename):
    stringarray = file_to_array(filename)
    digitarray = stringarray_to_digitarray(stringarray) #digitarray is 12x4 matrix, need to check if array is valid

    if np.unique(digitarray).size == len(digitarray):
        raise ValueError("Cards must be unique")

    if not len(digitarray) == 12:
        raise ValueError("There must be 12 cards")

    for j in digitarray:
        if not np.all(i ==0 or i == 1 or i == 2 for i in digitarray[j]):
            raise ValueError("Card must contain only digits 0,1,2")

    if not all(len(digitarray[i])==4 for i in digitarray):
        raise ValueError("Cards must contain only four digits")

    else:
        iterlist = list(itertools.combinations(digitarray, 3))

        count = 0

        for i in range(0, len(iterlist)):
            if isvalid((iterlist)[i]):
                count += 1

    return count

print(countvalid('/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards.txt'))

#string.format for object oriented programming
