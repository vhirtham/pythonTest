def MyFunc(bool):
    if bool:
        print("The answer is 42")
    else:
        print("This branch is not covered")


def test_PrintHelloWorld():
    print('Hello world')

def test_MyFunc():
    MyFunc(True)