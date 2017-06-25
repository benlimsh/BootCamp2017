#Problem 4
import box as bx
import sys
import random
import itertools

def psl(list1, x):
    count=1
    for i in range(len(list1)):
        for c in itertools.combinations(list1, i):
            if sum(c) == x:
                count=count+1
    if count>1:
        return True
    else:
        return False

def shutthebox():
    if len(sys.argv)!=2:
        name=input("Enter your name:")
    else:
        name=argv[1]

    numbers_two=range(2,12,1)
    numbers_one=range(1,6,1)
    roll=random.choice(numbers_two)
    remaining=list(range(1, 10, 1))


    while bx.isvalid(roll,remaining):
        if sum(remaining)<=6:
            roll=random.choice(numbers_one)
        else:
            roll=random.choice(numbers_two)

        print("Numbers left:", remaining)
        print("Roll:", roll)
        score=sum(remaining)
        if not psl(remaining,roll):
            print("Game over!")
            print("Score for player:", name, ":", score ,"points")
        eliminate=input("Numbers to eliminate: ")

        a=bx.parse_input(eliminate,remaining)
        sum_a=sum(a)

        if sum_a==roll and (not a)==False and set(a)<set(remaining) and psl(remaining,sum_a):
            remaining = [x for x in remaining if x not in a]

        else:
            print("Invalid input")

    if not remaining:
        print("Score for player", name,  ":", score, "points")
        print("Congratulations!! You shut the box!")

    return

shutthebox()
