# test_chebyshev
#
#

import matplotlib.pyplot as plt
__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import numpy as np
import scipy.special as sc
import reikna.cluda as cluda

BACKEND = 'cuda'

if BACKEND == 'opencl':
  api = cluda.ocl_api()
elif BACKEND == 'cuda':
  api = cluda.cuda_api()
else:
  raise ValueError(f'Backend can only be `cuda` or `opencl`. Tried {BACKEND}')

# Compile the library and the tester kernel
thr = api.Thread.create()



prog = thr.compile(open('/home3/marcos.romero/py-lib99ocl/tests/kronrod.c').read(),
compiler_options=[f"-I/home3/marcos.romero/py-lib99ocl/lib99ocl/src"],
)


a = np.float32(-0.0)
b = np.float32(+1.0)
integral = thr.to_device(np.float32([0.]))
prog.kernel_quadgk(integral, a, b, global_size=(1,), local_size=(1,))
print("Integral:", integral)

# vim: fdm=marker ts=2 sw=2 sts=2 sr et

