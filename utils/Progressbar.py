from copy import deepcopy

strs = ['━', '█']


def Progressbar(contents, str="progress"):
    origal = deepcopy(contents) if isinstance(contents, zip) else contents
    length = len(list(contents))  # 加入list()可以处理zip对象
    for index, content in enumerate(origal):
        percentage = int((index + 1) / length * 100)
        str1 = '━' * (percentage // 2)
        str2 = ' ' * (50 - percentage // 2)
        print("\r", end="")
        print(f"{str}: {str1}{str2}[{percentage}%][{index + 1} /{length}]", end="")
        yield content
    print("")


if __name__ == '__main__':
    a = range(2)
    for i in Progressbar(a):
        pass
