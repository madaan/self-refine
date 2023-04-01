COUNT_VAR_PROMPT = '''
"""CODE SNIPPET"""
import sys
# Reads input from terminal and returns it
def input(): 
    return sys.stdin.readline().strip()
# finds a largest perfect power which is smaller 
# than the input number
def resolve():
    # read input
    x=int(eval(input()))
    ans=0
    for i in range(1,33):
        for j in range(2,11):
            y=i**j
            if y<x:
                ans=max(y,ans)
            elif y==x:
                ans=max(y,ans)
                break
            else:
                break
    print(ans)
if __name__ == "__main__":
    resolve()
"""CODE SNIPPET END"""

All variables in the code snippet:
x - random
ans - prefix of answer
i - random
j - random
y - random


"""CODE SNIPPET"""
# ABC097B - Exponential
def main():
    X = int(eval(input()))
    cand = {{1}}
    for i in range(2, int(X ** 0.5) + 1):
        p = 2
        while i ** p <= X:
            cand.add(i ** p)
            p += 1
    ans = max(cand)
    print(ans)


if __name__ == "__main__":
    main()
"""CODE SNIPPET END"""

All variables in the code snippet:
X - random
cand - prefix of candidate
i - random
p - random
ans - prefix of answer


"""CODE SNIPPET"""
user_input = int(eval(input()))
max_power_result = 0

for number in range(user_input):
    for power in range(9):
        if((number+1)**(power+2)<=user_input):
            max_power_result = max(max_power_result, (number+1)**(power+2))

print(max_power_result)
"""CODE SNIPPET END"""

All variables in the code snippet:
user_input - combination of user and input
max_power_result - combination of max, power and result
number - word number
power - word power


"""CODE SNIPPET"""
{code}
"""CODE SNIPPET END"""

All variables in the code snippet:
'''.strip() + '\n'

PROMPT_CRITIQUE = """
I have some code. Can you give one suggestion to improve readability. Don't fix the code, just give a suggestion.

{code}
""".strip() + '\n'

PROMPT_FIX = """
I have some code. Can you give one suggestion to improve readability. Don't fix the code, just give a suggestion.

{code}

{suggestion}

Now fix the code.
""".strip() + '\n'