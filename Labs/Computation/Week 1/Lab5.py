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
        print ("Process interrupted at iteration.", i, ".")

    finally:
        return walk

#Problem 3 and 4
class ContentFilter():
    def __init__(self, name):
        if isinstance(name, str):
            file_name = name + ".txt"
            with open(name, 'r') as myfile:
                self.contents = myfile.readlines()
                self.contents = [line.rstrip("\n") for line in self.contents]
                self.name = name
        else:
            raise TypeError("Name must be a string.")

    def uniform(self, name, mode = "w", case = "upper"):
        if mode != "w" and mode != "a":
            raise ValueError("Mode must be w or a.")
        else:
            file_name = name + ".txt"
            with open(file_name, 'w') as outfile:
                if case == "upper":
                    for lines in self.contents:
                        outfile.write(lines.upper() + "\n")
                if case == "lower":
                    for lines in self.contents:
                        outfile.write(lines.lower() + "\n")
                else:
                    raise ValueError("Case must be upper or lower.")

    def reverse(self, name, mode = "w", unit = "line"):
        if mode != "w" and mode != "a":
            raise ValueError("Mode must be w or a.")
        else:
            file_name = name + ".txt"
            with open(file_name, mode) as outfile:
                if unit == "word":
                    for lines in self.contents:
                        words = lines.split()
                        reversewords = " ".join(words[::-1])
                        outfile.write(reversewords + "\n")
                elif unit == "line":
                    for lines in self.contents[::-1]:
                        outfile.write(lines + "\n")
                else:
                    raise ValueError("Unit must be word or line.")

    def transpose(self, name, mode = "w"):
        if mode != "w" and mode != "a":
            raise ValueError("Mode must be w or a.")
        file_name = name + ".txt"
        length = len(self.contents[0].split()) #number of words in first row
        for lines in self.contents:
            if len(lines.split()) != length:
                raise ValueError("Number of words in each line are not equal.")
        with open(name, mode) as outfile:
            for i in range(0, length):
                words = [lines.split()[i] for lines in self.contents] #first word of each row
                line = " ".join(words) #joins first word of each row into a new row of words
                outfile.write(line + "\n")

    def __str__(self):
        text = " ".join(self.contents)
        charcount = len(text)
        alphacount = sum(t.isalpha() for t in text)
        numcount = sum(t.isdigit() for t in text)
        spacecount = sum(t.isspace() for t in text)
        linecount = len(self.contents)

        return "Source File: \t" + self.name  + ".txt" + \
        "\nTotal Characters: \t" + str(charcount) + \
        "\nAlphabetic Characters: \t" + str(alphacount) + \
        "\nNumerical Characters: \t" + str(numcount) + \
        "\nWhitespace Characters: \t" + str(spacecount) + \
        "\nNumber of Lines: \t" + str(linecount)
