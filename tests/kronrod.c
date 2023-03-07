#ifndef _GAUSS_KRONROD_C_
#define _GAUSS_KRONROD_C_

#define USE_DOUBLE 0
#define QUADRATURE_EPS 1e-14

#include <lib99ocl/core.c>
#include <lib99ocl/core.h>

WITHIN_KERNEL ftype custom_f(ftype x) {
  // return x*x;
  // return exp(-x * x / 0.01);
  return cos(log(x) / x) / x;
}

WITHIN_KERNEL ftype proxy_gauss_kronrod_quad(ftype a, ftype b, int N) {
  // int stack[1024];
  printf("Evaluation :: %d\n", N);
  // TODO: n adn w should be function arguments {{{
  // Nodes and weights for Gauss-Kronrod (7, 15)
  const ftype n15[15] = {-0.9914553711208126, -0.9491079123427585,
                         -0.8648644233597691, -0.7415311855993944,
                         -0.5860872354676911, -0.4058451513773972,
                         -0.2077849550078985, 0.0,
                         0.2077849550078985,  0.4058451513773972,
                         0.5860872354676911,  0.7415311855993944,
                         0.8648644233597691,  0.9491079123427585,
                         0.9914553711208126};

  const ftype n7[7] = {
      -0.9491079123427585, -0.7415311855993944, -0.4058451513773972, 0.0,
      0.4058451513773972,  0.7415311855993944,  0.9491079123427585};

  const ftype w15[15] = {
      0.0229353220105292, 0.0630920926299785, 0.1047900103222502,
      0.1406532597155259, 0.1690047266392679, 0.1903505780647854,
      0.2044329400752989, 0.2094821410847278, 0.2044329400752989,
      0.1903505780647854, 0.1690047266392679, 0.1406532597155259,
      0.1047900103222502, 0.0630920926299785, 0.0229353220105292};
  const ftype w7[7] = {0.1294849661688697, 0.2797053914892767,
                       0.3818300505051189, 0.4179591836734694,
                       0.3818300505051189, 0.2797053914892767,
                       0.1294849661688697};
  // }}}

  ftype Q7 = 0;
  ftype x7[7] = {0};
  ftype Q15 = 0;
  ftype x15[15] = {0};

  for (int i = 0; i < 7; i++) {
    x7[i] = 0.5 * ((b - a) * n7[i] + b + a);
    Q7 += w7[i] * custom_f(x7[i]);
  }
  Q7 *= (b - a) / 2;

  for (int i = 0; i < 15; i++) {
    x15[i] = 0.5 * ((b - a) * n15[i] + b + a);
    Q15 += w15[i] * custom_f(x15[i]);
  }
  Q15 *= (b - a) / 2;

  if (N > 0) {
    return Q15;
  }

  if ((Q7 > 1e300) || (Q15 > 1e300)) {
    // warning("Infinite or NA function value encountered.")
    return Q15;
  } else if (fabs(Q15 - Q7) < QUADRATURE_EPS) {
    return Q15;
  } else if (fabs(b - a) < 16 * QUADRATURE_EPS) {
    return proxy_gauss_kronrod_quad((a + b) / 2, b, N + 1); // aka. Q2
  } else {
    return proxy_gauss_kronrod_quad((a + b) / 2, b, N + 2) +
           proxy_gauss_kronrod_quad(a, (a + b) / 2, 0);
  }
}

WITHIN_KERNEL ftype quadgk(ftype a, ftype b) {
  if (a == b) {
    return 0;
  } else if (a > b) {
    return -proxy_gauss_kronrod_quad(b, a, 0);
  }

  return proxy_gauss_kronrod_quad(a, b, 0);
}

KERNEL void kernel_quadgk(GLOBAL_MEM ftype *answer, ftype a, ftype b) {
  const int idx = get_global_id(0);
  answer[idx] = quadgk(a, b);
}

#endif // _GAUSS_KRONROD_C_

// vim: fdm=marker ts=2 sw=2 sts=2 sr noet
