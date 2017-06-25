#Problem 1

def prob1(list):
    list=[min(list),max(list),sum(list)/(len(list)*1.0)]
    return list

#Problem 2

def prob2():

    num1=1
    num2=num1
    num2+=1

    word1="word"
    word2=word1
    word2+='a'

    list1=[4,5,6]
    list2=list1
    list2.append(1)

    tuple1=('physics', 'chemistry')
    tuple2=tuple1
    tuple2+=(1,)

    dict1 = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}
    dict2=dict1
    dict2[1]='a'

    result=[num2==num1,word2==word1,list2==list1,tuple2==tuple1,dict2==dict1]
    return result

print(prob2())

#Problem 3
import calculator as calc
def hypotenuse(x,y):
    result=calc.sqroot(calc.producttwo(x,x)+calc.producttwo(y,y))
    return result

#Problem 4
import box as bx
import sys
import random

def shutthebox():
    if len(sys.argv)!=2:
        name=input("Enter your name:")
    else:
        name=argv[1]

    roll=random.choice(numbers_two)
    remaining=list(range(1, 9, 1))
    numbers_two=range(2,12,1)
    numbers_one=range(1,6,1)

    while isvalid(roll,remaining):
        if sum(remaining)<=6:
            roll=random.choice(numbers_one)
        else:
            roll=random.choice(numbers_two)

        print("Numbers left: %d") %remaining
        print("Roll: %d") %roll
        eliminate=input("Numbers to eliminate: ")
        a=parse_input(eliminate,remaining)

        if not a:
            print("Invalid input")
        else:
            remaining = [x for x in remaining if x not in a]

    score=sum(remaining)
    if not remaining:
        print("Score for player %s  :  %d points") %(name, score)
        print("Congratulations!! You shut the box!")

    else:
        print("Game over!")
        print("Score for player %s  :  %d points") %(name,score)
