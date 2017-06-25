# test_solutions.py
"""Volume 1B: Testing.
<Name>
<Class>
<Date>
"""
import numpy as np
import solutions as soln
import pytest
from solutions import addition
from solutions import smallest_factor
from solutions import operator
from solutions import ComplexNumber
from solutions import file_to_array
from solutions import number_to_digits
from solutions import stringarray_to_digitarray
from solutions import sumcolumn
from solutions import isvalid
from solutions import countvalid

# Problem 1: Test the addition and fibonacci functions from solutions.py
def test_addition():
    assert addition(1,3) == 4
    assert addition(-5,-7) == -12
    assert addition(-6,14) == 8

def test_smallest_factor():
    assert smallest_factor(1) == 1
    assert smallest_factor(2) == 2
    assert smallest_factor(7) == 7
    assert smallest_factor(14) == 2

# Problem 2: Test the operator function from solutions.py
def test_operator():
    assert operator(1,2,"+") == 3
    assert operator(1,2,"-") == -1
    assert operator(1,2,"*") == 2
    assert operator(1,2,"/") == 0.5

    with pytest.raises(Exception) as excinfo:
        operator(1,2,3)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Oper should be a string"

    with pytest.raises(Exception) as excinfo:
        operator(1,2,"++")
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Oper should be one character"

    with pytest.raises(Exception) as excinfo:
        operator(1,0,"/")
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "You can't divide by zero!"

    with pytest.raises(Exception) as excinfo:
        operator(1,2,"b")
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Oper can only be: '+', '/', '-', or '*'"

# Problem 3: Finish testing the complex number class
@pytest.fixture

def set_up_complex_nums():
    number_1 = soln.ComplexNumber(1, 2)
    number_2 = soln.ComplexNumber(5, 5)
    number_3 = soln.ComplexNumber(2, 9)
    number_4 = soln.ComplexNumber(0, 0)
    number_5 = soln.ComplexNumber(3, -4)
    return number_1, number_2, number_3, number_4, number_5

def test_conjugate(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert ComplexNumber.conjugate(number_1) == soln.ComplexNumber(1,-2)
    assert ComplexNumber.conjugate(number_4) == soln.ComplexNumber(0,0)

def test_norm(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert ComplexNumber.norm(number_4) == soln.ComplexNumber(0,0)
    assert ComplexNumber.norm(number_5) == soln.ComplexNumber(5,0)

def test_complex_addition(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert number_1 + number_2 == soln.ComplexNumber(6, 7)
    assert number_1 + number_3 == soln.ComplexNumber(3, 11)
    assert number_2 + number_3 == soln.ComplexNumber(7, 14)
    assert number_3 + number_3 == soln.ComplexNumber(4, 18)

def test_complex_multiplication(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert number_1 * number_2 == soln.ComplexNumber(-5, 15)
    assert number_1 * number_3 == soln.ComplexNumber(-16, 13)
    assert number_2 * number_3 == soln.ComplexNumber(-35, 55)
    assert number_3 * number_3 == soln.ComplexNumber(-77, 36)

def test_complex_subtraction(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert number_1 - number_2 == soln.ComplexNumber(-4, -3)
    assert number_1 - number_3 == soln.ComplexNumber(-1, -7)
    assert number_2 - number_3 == soln.ComplexNumber(3, -4)
    assert number_3 - number_3 == soln.ComplexNumber(0, 0)

def test_complex_truediv(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert number_1 / number_2 == soln.ComplexNumber(3/10*1.0, 1/10*1.0)
    assert number_1 / number_3 == soln.ComplexNumber(4/17*1.0, -1/17*1.0)
    assert number_2 / number_3 == soln.ComplexNumber(11/17*1.0, -7/17*1.0)
    assert number_3 / number_3 == soln.ComplexNumber(1, 0)
    with pytest.raises(Exception) as excinfo:
        number_1 / number_4
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Cannot divide by zero"

def test_eq(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert (number_1 == number_1) == True
    assert (number_1 == number_2) == False

def test_str(set_up_complex_nums):
    number_1, number_2, number_3, number_4, number_5 = set_up_complex_nums
    assert str(number_1) == "1+2i"
    assert str(number_2) == "5+5i"
    assert str(number_3) == "2+9i"
    assert str(number_4) == "0+0i"
    assert str(number_5) == "3-4i"

# Problem 4: Write test cases for the Set game.
'''
each card is represented as a 4 bit integer in base 3
because there are 3 possible options for each feature
{}{}{}{}={quantity}{shape}{color}{pattern}

quantity:
    0: one
    1: two
    2: three
shape:
    0: diamond
    1: oval
    2: squiggly
color:
    0: green
    1: red
    2: purple
pattern:
    0: blank
    1: solid
    2: stripes
'''
@pytest.fixture
def set_up_files():
    file_1 = '/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards1.txt'
    file_2 = '/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards2.txt'
    file_3 = '/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards3.txt'
    file_4 = '/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards4.txt'
    file_5 = '/Users/benjaminlim/Desktop/OSM Lab/Prob Sets/Computation/Week 1/cards5.txt'

    return file_1, file_2, file_3, file_4, file_5

def test_countvalid(set_up_files):
    file_1, file_2, file_3, file_4, file_5 = set_up_files

    assert countvalid(file_1) == 6

    with pytest.raises(Exception) as excinfo:
        countvalid(file_2)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Card must contain only digits 0,1,2"

    with pytest.raises(Exception) as excinfo:
        countvalid(file_3)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "There must be 12 cards"

    with pytest.raises(Exception) as excinfo:
        countvalid(file_4)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Cards must contain only four digits"

    with pytest.raises(Exception) as excinfo:
        countvalid(file_5)
    assert excinfo.typename == 'ValueError'
    assert excinfo.value.args[0] == "Cards must be unique"
