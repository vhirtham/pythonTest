def MyFunc(bool):
    if bool:
        print("The answer is 42")
    else:
        print("This branch is not covered")

def AddNumbers(a,b):
    if not (isinstance(a, int) or isinstance(a, float)):
        raise TypeError("First argument must be a integer or float")
    if not (isinstance(b, int) or isinstance(b, float)):
        raise TypeError("Second argument must be a integer or float")

    return a + b