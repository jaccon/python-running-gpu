from numba import jit, cuda
from timeit import default_timer as timer
import numpy as np

# run on cpu
def func(a):								
	for i in range(100000000):
		a[i]+= 1	

# run on gpu
@jit(target_backend='cuda')						
def func2(a):
	for i in range(100000000):
		a[i]+= 1
if __name__=="__main__":
	n = 100000000							
	a = np.ones(n, dtype = np.float64)
	
	start = timer()
	func(a)
	print("Running in CPU:", timer()-start)	
	
	start = timer()
	func2(a)
	print("Running in GPU:", timer()-start)
