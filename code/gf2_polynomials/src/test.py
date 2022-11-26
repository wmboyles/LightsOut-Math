import gf2_polynomial

from time import time
if __name__ == '__main__':
    start = time()
    gf2_polynomial.grid_nullities(10_000)
    print(time() - start)
