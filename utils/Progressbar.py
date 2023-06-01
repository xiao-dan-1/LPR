strs = ['━', '█']


def Progressbar(contents, str="progress"):
    length = len(contents)
    epoch = length - 1
    for index, content in enumerate(contents):
        percentage = int(index / epoch * 100)
        str1 = '━' * (percentage // 2)
        str2 = ' ' * (50 - percentage // 2)
        print("\r", end="")
        print(f"{str}: {str1}{str2}[{percentage}%][{index + 1} /{length}]", end="")
        yield content


if __name__ == '__main__':
    a = range(2)
    for i in Progressbar(a):
        pass