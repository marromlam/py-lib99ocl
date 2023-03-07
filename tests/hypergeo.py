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
//#include <lib99ocl/core.c>
//#include <lib99ocl/complex.c>
//#include <lib99ocl/special.c>
//#include <lib99ocl/cspecial.c>
//#include <lib99ocl/lineshapes.c>
//#include <lib99ocl/hypergeom.c>
//#include <lib99ocl/hypergeom2.c>
#include <lib99ocl/special/gamma.c>
#include <lib99ocl/special/psi.c>
#include <lib99ocl/core.c>
#include <lib99ocl/special/hyp2f1.c>


WITHIN_KERNEL
ftype patata(const ftype x)
{
  // printf("%f\\n", x);
  return (x <20000) ? patata(x+1) : 2;
}


KERNEL void
kernel_hyp2f1(GLOBAL_MEM ftype *ans, GLOBAL_MEM ftype *z,
       const ftype a, const ftype b, const ftype c)
{
  const SIZE_T i = get_global_id(0);
  ans[i] = hyp2f1(a, b, c, z[i]);
}

""", compiler_options=[f"-I/home3/marcos.romero/py-lib99ocl/lib99ocl/src"])


chebyshev_h = sc.hyp2f1
chebyshev_d = prog.kernel_hyp2f1

a, b, c = 1, 1, 10
xmin = -1
xmax = 7
xh = np.linspace(xmin, xmax, 100000)

xd = thr.to_device(xh).astype(np.float64)
# print(xd)
yd = thr.empty_like(xd)

chebyshev_d(yd, xd, np.float64(a), np.float64(b), np.float64(c),
            global_size=(len(xh),))
ans_h = np.nan_to_num(chebyshev_h(a, b, c, xh))
print(ans_h)
ans_d = np.nan_to_num(yd.get())
print(ans_d)

print(f"Max difference: {np.max(ans_d - ans_h)}")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
