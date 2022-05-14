# test_chebyshev
#
#

__all__ = []
__author__ = ["name"]
__email__ = ["email"]


import numpy as np
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
#include <lib99ocl/lineshapes.c>

KERNEL void
kernel_chebyshev(GLOBAL_MEM ftype *ans, const GLOBAL_MEM ftype *x,
       const GLOBAL_MEM ftype *t, const int deg, const ftype xL,
       const ftype xH)
{
  const SIZE_T i = get_global_id(0);
  ans[i] = chebyshev(x[i], t, deg, xL, xH);
}

KERNEL void
kernel_chebyshevIntegral(GLOBAL_MEM ftype *ans, const GLOBAL_MEM ftype *x,
       const GLOBAL_MEM ftype *t, const int deg, const ftype xL,
       const ftype xH)
{
  const SIZE_T i = get_global_id(0);
  ans[i] = chebyshevIntegral(x[i], t, deg, xL, xH);
}
""", compiler_options=[f"-Ilib99ocl/src"])


chebyshev_h = np.polynomial.chebyshev.chebval
chebyshev_d = prog.kernel_chebyshev
chebyshevIntegral_d = prog.kernel_chebyshevIntegral

xmin = -1
xmax = 1
# xh = np.random.randn(N).astype(numpy.float64)
xh = np.linspace(xmin, xmax, 2000000).astype(np.float64)

xd = thr.to_device(xh)
yd = thr.empty_like(xd)

th = np.float64([2., 3.4545, 5., 9., 0.3])
td = thr.to_device(th).astype(np.float64)

deg = len(th) - 1

chebyshev_d(yd, xd, td, np.int32(deg), np.float64(xmin), np.float64(xmax),
            global_size=(len(xh),))
ans_d = yd.get()
ans_h = chebyshev_h(xh, th)
print(ans_d)
print(ans_h)

print(f"Max difference: {np.max(ans_d - ans_h)}")
print(np.trapz(ans_h, xh))
chebyshevIntegral_d(yd, xd, td, np.int32(deg), np.float64(xmin), np.float64(xmax),
            global_size=(len(xh),))
ans_d = yd.get()
print(ans_d[-1] - ans_d[0])


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
