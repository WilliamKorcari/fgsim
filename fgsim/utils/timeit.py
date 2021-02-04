def timeit(func, n=1):
    import time

    def wrapper(*arg, **kw):
        """source: http://www.daniweb.com/code/snippet368.html"""
        t1 = time.time()
        for i in range(n):
            res = func(*arg, **kw)
        t2 = time.time()
        print(func.__name__, (t2 - t1) / n)
        return res

    return wrapper
