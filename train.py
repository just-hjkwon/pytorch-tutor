import functools


print = functools.partial(print, flush=True)


def main():
    print('Hello!')


if __name__ == '__main__':
    main()