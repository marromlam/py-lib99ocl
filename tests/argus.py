# test_argus
#
#

__all__ = []
__author__ = ["Marcos Romero Lamas"]
__email__ = ["mromerol@cern.ch"]


import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sc
import reikna.cluda as cluda

BACKEND = 'opencl'

if BACKEND == 'opencl':
  api = cluda.ocl_api()
elif BACKEND == 'cuda':
  api = cluda.cuda_api()
else:
  raise ValueError(f'Backend can only be `cuda` or `opencl`. Tried {BACKEND}')


m0 = 5067
p = 0.5
c = -7

xLL = 4800
xUL = 6000


# Compile the library and the tester kernel
thr = api.Thread.create()
prog = thr.compile(f"""
#define USE_DOUBLE 1
// #include <lib99ocl/core.c>
// #include <lib99ocl/complex.c>
// #include <lib99ocl/special.c>
// #include <lib99ocl/cspecial.c>
// #include <lib99ocl/lineshapes.c>
// #include <lib99ocl/hypergeom.c>
//#include <lib99ocl/hypergeom2.c>
#include <exposed/kernels.ocl>

KERNEL void
tester(GLOBAL_MEM ftype *ans, GLOBAL_MEM ftype *z)
{{
  const SIZE_T i = get_global_id(0);
  const ftype num = argus(z[i], {m0}, {c}, {p});
  const ftype den = argus_integral({xLL}, {xUL}, {m0}, {c}, {p});
  ans[i] = num/den;
}}

""", compiler_options=[f"-I/home3/marcos.romero/py-lib99ocl/lib99ocl/src"])


xh = np.linspace(xLL, xUL, int(1e4))

xd = thr.to_device(xh).astype(np.float64)
yd = thr.empty_like(xd)

prog.tester(yd, xd, global_size=(len(xh),))
ans_d = np.nan_to_num(yd.get())

plt.plot(xh, ans_d, label=f"c={c}  p={p} m0={m0}")

prog.kernel_argus(yd, xd, np.float64(m0), np.float64(c), np.float64(p), np.float64(xLL), np.float64(xUL), global_size=(len(xh),))
ans_d = np.nan_to_num(yd.get())
plt.plot(xh, ans_d, label=f"nueva")

prog.py_argus(yd, xd, np.float64(m0), np.float64(c), np.float64(p), global_size=(len(xh),))
ans_d = np.nan_to_num(yd.get())
norm = np.trapz(ans_d, xh)
plt.plot(xh, ans_d/norm, '-.', label="vieja")

plt.legend()
plt.show()
print(np.trapz(ans_d, xh))

# print(f"Max difference: {np.max(np.abs(ans_d - ans_h))}")


# vim: fdm=marker ts=2 sw=2 sts=2 sr et
