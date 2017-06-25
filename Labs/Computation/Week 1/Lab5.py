#Problem 1
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def arithmagic():
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")

    if len(step_1)!=3 or is_number(step_1)==False:
        raise ValueError("Input must be a 3-digit number")
    if abs(int(step_1[0])-int(step_1[2])) < 2:
        raise ValueError("First and last digits must differ by at least 2")

    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")

    if step_2 != step_1[::-1]:
        raise ValueError("Second number must be reverse of first number")

    step_3 = input("Enter the positive difference of these numbers: ")

    if int(step_3) != abs(int(step_1)-int(step_2)):
        raise ValueError("Third number must be positive difference of the first two numbers")

    step_4 = input("Enter the reverse of the previous result: ")

    if step_4 != step_3[::-1]:
        raise ValueError("Fourth number must be reverse of third number")

    print(str(step_3) + " + " + str(step_4) + " = 1089 (ta-da!)")

arithmagic()

#Problem 2
from random import choice
def random_walk(max_iters=1e12):
    walk = 0
    direction = [1, -1]
    try:
        for i in range(int(max_iters)):
            walk += choice(direction)
        print ("Process completed.")

    except KeyboardInterrupt:
        print ("Process interrupted at iteration", i, ".")

    finally:
        return walk


#Problem 3 and 4
class ContentFilter():
    def __init__(self, name):
        if (isinstance(name, str)):
            self.name = name
            with open(name, 'r') as myfile:
                contents = myfile.readlines()
        else:
            raise TypeError("Name must be a string.")

    def uniform(self, name, case = "upper"):
        with open(name, 'w') as outfile:
            if case == "upper":
                with open(name, 'r') as outfile:
                    lines = outfile.readlines()
                with open(name, 'w') as outfile:
                    for line in lines:
                        outfile.write(line.upper())

            if case == "lower":
                with open(name, 'r') as outfile:
                    lines = outfile.readlines()
                with open(name, 'w') as outfile:
                    for line in lines:
                        outfile.write(line.lower())
            else:
                raise ValueError("Case must be upper or lower.")

    def reverse(self, name, unit = "line"):
        with open(name, 'w') as outfile:
            if unit == "word":
                with open(name, 'r') as outfile:
                    lines = outfile.readlines()
                with open(name, 'w') as outfile:
                    for line in lines:
                        for word in line.split():
                            outfile.write(word[::-1])

            if unit == "line":
                with open(name, 'r') as outfile:
                    lines = outfile.readlines()
                with open(name, 'w') as outfile:
                    for line in reversed(lines):
                        outfile.write(line)
            else:
                raise ValueError("Unit must be word or line")

'''
    def transpose(self, name):
        pass

'''
