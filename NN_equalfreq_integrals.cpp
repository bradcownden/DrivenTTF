#include <stdio.h>
#include <math.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_sf_hyperg.h>
#include <gsl/gsl_sf_gamma.h>
#include <gsl/gsl_deriv.h>

#define pi M_PI
#define d 3.0
#define m 2.0
#define Delta ((d + sqrt(pow(d, 2.0) + 4. * pow(m, 2.))) / 2.)


// Structure for NN basis functions
struct nn_params {
  // Frequency and Delta are required
  double w;
  double delta;
};

// Compute the NN basis function
double nn(double x, void *p) {
  // Read parameters
  nn_params &params = *reinterpret_cast<nn_params *>(p);
  double w = params.w;
  double delta = params.delta;
  // Compute a, b, c out of delta and w
  double a = (delta + w) / 2.;
  double b = (delta - w) / 2.;
  double c = d / 2.;

  // Return the NN basis function
  double f = pow(cos(x), delta) * gsl_sf_hyperg_2F1(a, b, c, pow(sin(x), 2.));
  return f;
} 
  
int main (void) {

  // Test finding the derivative of a function at a point
  printf("Derivatives of NN basis functions at various points:\n");

  gsl_function F;
  double result, error;

  nn_params params;
  params.w = M_PI / 2.;
  params.delta = Delta;

  F.function = &nn;
  F.params = reinterpret_cast<void *>(&params);
  
  for (int i = 0; i <10; i++) {
    int code = gsl_deriv_backward(&F, M_PI * (float(i) / 10.) / 2.,
				      1.0e-8, &result, &error);
    if (code) {
      printf("ERROR in derivative\n");
    } else {
      printf("%.12f\n", result);
    }
  }



  int npts = 3;
  double pts[npts];
  pts[0] = 0.;
  pts[1] = M_PI / 2.;
  pts[2] = M_PI / 2.;
  
  // Test integration using gsl QAGS
  int N = 1e8;
  gsl_integration_workspace * wksp =
    gsl_integration_workspace_alloc(N);

  int integral = gsl_integration_qagp(&F, pts, npts, 0., 1.e-7,
		       N, wksp, &result, &error);
  if (integral) {
    printf("ERROR in integration: %s\n", gsl_strerror(integral));
  }

  printf("result          = %.18f\n", result);
  printf("estimated error = %.18f\n", error);
  printf("intervals       = %zu\n", wksp->size);

  gsl_integration_workspace_free(wksp);
  
  
  return 0;

}
