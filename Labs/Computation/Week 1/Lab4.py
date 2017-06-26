#Problem 1
import math

class Backpack(object):

    def __init__(self,name,color,max_size=5):
        self.name = name
        self.color = color
        self.contents = []
        self.max_size = max_size

    def put(self,item):
        if len(self.contents) == self.max_size:
            print("No Room!")
        else:
            self.contents.append(item)

    def take(self,item):
        self.contents.remove(item)

    def dump(self):
        self.contents = []

def test_backpack():
    testpack = Backpack("Barry", "black")
    if testpack.max_size != 5:
        print("Wrong default max_size!")
    for item in ["pencil", "pen", "paper", "computer"]:
        testpack.put(item)
    print(testpack.contents)

    for item in ["ruler","eraser"]:
        testpack.put(item)
    print(testpack.contents)

    testpack.take("ruler")
    print(testpack.contents)

    testpack.dump()
    print(testpack.contents)

test_backpack()

#Problem 2

class Jetpack(Backpack):

    def __init__(self,name,color,max_size=2,fuel=10):
        Backpack.__init__(self,name,color,max_size)
        self.fuel = fuel

    def fly(self, fuel):
        if self.fuel < fuel:
            print("Not enough fuel!")
        else:
            self.fuel = self.fuel - fuel

    def dump(self):
        self.fuel = 0
        self.contents = []

def test_jetpack():
    testjetpack = Jetpack("Barry", "black")
    if testjetpack.max_size != 2:
        print("Wrong default max_size!")
    if testjetpack.fuel != 10:
        print("Wrong default fuel!")
    for item in ["pencil", "pen"]:
        testjetpack.put(item)
    print(testjetpack.contents)

    for item in ["ruler"]:
        testjetpack.put(item)
    print(testjetpack.contents)

    testjetpack.take("pen")
    print(testjetpack.contents)

    testjetpack.fly(5)
    print(testjetpack.fuel)

    testjetpack.fly(6)
    print(testjetpack.fuel)

    testjetpack.dump()
    print(testjetpack.contents)
    print(testjetpack.fuel)

test_jetpack()

#Problem 3
class Backpack(object):

    def __init__(self,name,color,max_size=5):
        self.name = name
        self.color = color
        self.contents = []
        self.max_size = max_size

    def put(self,item):
        if len(self.contents) == self.max_size:
            print("No Room!")
        else:
            self.contents.append(item)

    def take(self,item):
        self.contents.remove(item)

    def dump(self):
        self.contents = []

    def __eq__(self,other):
        return (self.name == other.name and self.color == other.color and len(self.contents) == len(other.contents))

    def __str__(self):
        return "Owner:\t\t"+ self.name + \
        "\nColor:\t\t" + self.color + \
        "\nSize:\t\t"+ str(len(self.contents)) + \
        "\nMax Size: \t" + str(self.max_size) + "\nContents:\t" + \
        str(self.contents)

def test_magic():
    magicpack = Backpack("Johnny", "black")
    otherpack1 = Backpack("Jenny", "black")
    otherpack2 = Backpack("Johnny", "black")

    print(magicpack==otherpack1)
    print(magicpack==magicpack)

    for item in ("dog","cat","barabara streisand"):
        magicpack.put(item)

    print(str(magicpack))

test_magic()

#Problem 4
class ComplexNumber(object):
    def __init__(self, real, imag):
        self.real = real
        self.imag = imag

    def conjugate(self):
        return ComplexNumber(self.real, -self.imag)

    def __abs__(self):
        return math.sqrt(self.real**2 + self.imag**2)

    def __lt__(self, other):
        return abs(self) < abs(other)

    def __gt__(self, other):
        return abs(self) > abs(other)

    def __eq__(self, other):
        return self.imag == other.imag and self.real == other.real

    def __ne__(self,other):
        return not self == other

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

def operations_test():
    a = ComplexNumber(1,1)
    b = ComplexNumber(2,2)

    print (abs(a) == 2**0.5)
    print ((a < b) == True)
    print ((a > b) == False)
    print ((a==b) == False)
    print ((a!=b) == True)
    print (a+b == ComplexNumber(3,3))
    print (a-b == ComplexNumber(-1,-1))
    print (a*b == ComplexNumber(0,4))
    print (a/b == ComplexNumber(0.5,0))

operations_test()
