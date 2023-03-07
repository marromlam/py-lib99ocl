# test_chebyshev
#
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import numpy as np
import scipy.special as sc
import reikna.cluda as cluda

BACKEND = 'opencl'

if BACKEND == 'opencl':
  api = cluda.ocl_api()
elif BACKEND == 'cuda':
  api = cluda.cuda_api()
else:
	raise ValueError(f'Backend can only be `cuda` or `opencl`. Tried {BACKEND}')

# Compile the library and the tester kernel
thr = api.Thread.create()
prog = thr.compile("""
#define USE_DOUBLE 1
#include <lib99ocl/core.c>
#include <lib99ocl/complex.c>
#include <lib99ocl/special.c>
#include <lib99ocl/cspecial.c>
#include <lib99ocl/lineshapes.c>
#include <lib99ocl/hypergeom.c>
//#include <lib99ocl/hypergeom2.c>

KERNEL void
kernel_psi(GLOBAL_MEM ftype *ans, GLOBAL_MEM ftype *z)
{
  const SIZE_T i = get_global_id(0);
  ans[i] = digamma(z[i]);
}

""", compiler_options=[f"-I/home3/marcos.romero/py-lib99ocl/lib99ocl/src"])


chebyshev_h = sc.psi
chebyshev_d = prog.kernel_psi

xmin = -30
xmax = 30
xh = np.linspace(xmin, xmax, int(1e6))

xd = thr.to_device(xh).astype(np.float64)
print(xd)
yd = thr.empty_like(xd)

chebyshev_d(yd, xd, global_size=(len(xh),))
ans_d = np.nan_to_num(yd.get())
ans_h = np.nan_to_num(chebyshev_h(xh))
print(ans_d)
print(ans_h)

print(f"Max difference: {np.max(np.abs(ans_d - ans_h))}")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
