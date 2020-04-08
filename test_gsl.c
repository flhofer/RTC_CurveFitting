#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include <locale.h>
#include <gsl/gsl_histogram.h>

#include <time.h>			// constants and functions for clock
#include <errno.h>			// system error management (LIBC)
#include <string.h>			// strerror print

#define USEC_PER_SEC		1000000
#define NSEC_PER_SEC		1000000000

struct data
{
  double *t;
  double *y;
  size_t n;
};

/* model function: a * exp( -1/2 * [ (t - b) / c ]^2 ) */
static double
gaussian(const double a, const double b, const double c, const double t)
{
  const double z = (t - b) / c;
  return (a * exp(-0.5 * z * z));
}

/*
 *  func_f(): function to calculate fitting for, Gaussian fit
 *
 *  Arguments: - model fitting parameter vector
 * 			   - function data points, f(t) = y (real)
 * 			   - error/residual vector
 *
 *  Return value: status, success or error in computing value
 */
static int
func_f (const gsl_vector * x, void *params, gsl_vector * f)
{
  struct data *d = (struct data *) params;
  double a = gsl_vector_get(x, 0);
  double b = gsl_vector_get(x, 1);
  double c = gsl_vector_get(x, 2);
  size_t i;

  for (i = 0; i < d->n; ++i)
    {
      double ti = d->t[i];
      double yi = d->y[i];
      double y = gaussian(a, b, c, ti);

      gsl_vector_set(f, i, yi - y);
    }

  return GSL_SUCCESS;
}

/*
 *  func_df(): function to calculate differential vector, Gaussian fit
 *
 *  Arguments: - model fitting parameter vector
 * 			   - function data points, f(t) = y (real)
 * 			   - Jacobian matrix, first derivatives
 *
 *  Return value: status, success or error in computing value
 */
static int
func_df (const gsl_vector * x, void *params, gsl_matrix * J)
{
  struct data *d = (struct data *) params;
  double a = gsl_vector_get(x, 0);
  double b = gsl_vector_get(x, 1);
  double c = gsl_vector_get(x, 2);
  size_t i;

  for (i = 0; i < d->n; ++i)
    {
      double ti = d->t[i];
      double zi = (ti - b) / c;
      double ei = exp(-0.5 * zi * zi);

      gsl_matrix_set(J, i, 0, -ei);
      gsl_matrix_set(J, i, 1, -(a / c) * ei * zi);
      gsl_matrix_set(J, i, 2, -(a / c) * ei * zi * zi);
    }

  return GSL_SUCCESS;
}

/*
 *  func_fvv(): function to calculate quadratic error, Gaussian fit
 *
 *  Arguments: - model fitting parameter vector
 * 			   - velocity vector
 * 			   - function data points, f(t) = y (real)
 * 			   - d2/dx directional for geodesic acceleration
 *
 *  Return value: status, success or error in computing value
 */
static int
func_fvv (const gsl_vector * x, const gsl_vector * v,
          void *params, gsl_vector * fvv)
{
  struct data *d = (struct data *) params;
  double a = gsl_vector_get(x, 0);
  double b = gsl_vector_get(x, 1);
  double c = gsl_vector_get(x, 2);
  double va = gsl_vector_get(v, 0);
  double vb = gsl_vector_get(v, 1);
  double vc = gsl_vector_get(v, 2);
  size_t i;

  for (i = 0; i < d->n; ++i)
    {
      double ti = d->t[i];
      double zi = (ti - b) / c;
      double ei = exp(-0.5 * zi * zi);
      double Dab = -zi * ei / c;
      double Dac = -zi * zi * ei / c;
      double Dbb = a * ei / (c * c) * (1.0 - zi*zi);
      double Dbc = a * zi * ei / (c * c) * (2.0 - zi*zi);
      double Dcc = a * zi * zi * ei / (c * c) * (3.0 - zi*zi);
      double sum;

      sum = 2.0 * va * vb * Dab +
            2.0 * va * vc * Dac +
                  vb * vb * Dbb +
            2.0 * vb * vc * Dbc +
                  vc * vc * Dcc;

      gsl_vector_set(fvv, i, sum);
    }

  return GSL_SUCCESS;
}

/*
 *  callback(): callback function for the fitting iteration, Gaussian fit
 *  			Used for printing only
 *
 *  Arguments: - iteration number
 * 			   - function data points, f(t) = y (real)
 * 			   - GSL workspace reference
 *
 *  Return value: - none
 */
static void
callback(const size_t iter, void *params,
         const gsl_multifit_nlinear_workspace *w)
{
  gsl_vector *f = gsl_multifit_nlinear_residual(w);
  gsl_vector *x = gsl_multifit_nlinear_position(w);
  double avratio = gsl_multifit_nlinear_avratio(w);
  double rcond;

  (void) params; /* not used */

  /* compute reciprocal condition number of J(x) */
  gsl_multifit_nlinear_rcond(&rcond, w);

  fprintf(stderr, "iter %2zu: a = %.4f, b = %.4f, c = %.4f, |a|/|v| = %.4f cond(J) = %8.4f, |f(x)| = %.4f\n",
          iter,
          gsl_vector_get(x, 0),
          gsl_vector_get(x, 1),
          gsl_vector_get(x, 2),
          avratio,
          1.0 / rcond,
          gsl_blas_dnrm2(f));
}

/*
 *  solve_system(): solver setup and parameters, Gaussian fit
 *
 *  Arguments: - model fitting parameters
 * 			   - GSL fdf function parameters
 * 			   - GSL fdf parameters
 *
 *  Return value: - none
 */
static void
solve_system(gsl_vector *x, gsl_multifit_nlinear_fdf *fdf,
             gsl_multifit_nlinear_parameters *params)
{
  const gsl_multifit_nlinear_type *T = gsl_multifit_nlinear_trust;
  const size_t max_iter = 40;
  const double xtol = 1.0e-64;
  const double gtol = 1.0e-64;
  const double ftol = 1.0e-64;
  const size_t n = fdf->n;
  const size_t p = fdf->p;
  gsl_multifit_nlinear_workspace *work =
    gsl_multifit_nlinear_alloc(T, params, n, p);
  gsl_vector * f = gsl_multifit_nlinear_residual(work);
  gsl_vector * y = gsl_multifit_nlinear_position(work);
  int info;
  double chisq0, chisq, rcond;

  /* initialize solver */
  gsl_multifit_nlinear_init(x, fdf, work);

  /* store initial cost */
  gsl_blas_ddot(f, f, &chisq0);

  /* iterate until convergence */
  gsl_multifit_nlinear_driver(max_iter, xtol, gtol, ftol,
                              callback, NULL, &info, work);

  /* store final cost */
  gsl_blas_ddot(f, f, &chisq);

  /* store cond(J(x)) */
  gsl_multifit_nlinear_rcond(&rcond, work);

  gsl_vector_memcpy(x, y);

  /* print summary */
  fflush(stderr);
  fflush(stdout);
  fprintf(stderr, "NITER         = %zu\n", gsl_multifit_nlinear_niter(work));
  fprintf(stderr, "NFEV          = %zu\n", fdf->nevalf);
  fprintf(stderr, "NJEV          = %zu\n", fdf->nevaldf);
  fprintf(stderr, "NAEV          = %zu\n", fdf->nevalfvv);
  fprintf(stderr, "initial cost  = %.12e\n", chisq0);
  fprintf(stderr, "final cost    = %.12e\n", chisq);
  fprintf(stderr, "final x       = (%.12e, %.12e, %12e)\n",
          gsl_vector_get(x, 0), gsl_vector_get(x, 1), gsl_vector_get(x, 2));
  fprintf(stderr, "final cond(J) = %.12e\n", 1.0 / rcond);

  gsl_multifit_nlinear_free(work);
}


/*
/// tsnorm(): verifies timespec for boundaries + fixes it
///
/// Arguments: pointer to timespec to check
///
/// Return value: -
 *
 */
static inline void tsnorm(struct timespec *ts)
{
	while (ts->tv_nsec >= NSEC_PER_SEC) {
		ts->tv_nsec -= NSEC_PER_SEC;
		ts->tv_sec++;
	}
}

int
main (void)
{
  const size_t n = 100;  /* number of data points to fit */
  const size_t p = 3;    /* number of model parameters */

  gsl_vector *f = gsl_vector_alloc(n); /* data vector */
  gsl_vector *x = gsl_vector_alloc(p); /* model parameter vector */


  const double a = 10.0;  	  /* amplitude */ // = number of occurences
  const double b = 0.010500;  /* center */
  const double c = 0.000150;  /* width */
  const gsl_rng_type * T = gsl_rng_default;
  gsl_multifit_nlinear_fdf fdf;
  gsl_multifit_nlinear_parameters fdf_params =
    gsl_multifit_nlinear_default_parameters();

//  /*
//   * generate data for fitting test
//   */
//  struct data fit_data;
//  gsl_rng * r;
//  size_t i;
//
//  gsl_rng_env_setup ();
//  r = gsl_rng_alloc (T);
//
//  fit_data.t = malloc(n * sizeof(double));
//  fit_data.y = malloc(n * sizeof(double));
//  fit_data.n = n;
//
//  /* generate synthetic data with noise */
//  for (i = 0; i < n; ++i)
//    {
//	  /* Set sampling range 10ms +- 2ms */
//      double t = ((double)i / (double) n) // = (0..1)
//    		  * 0.002000 + 0.009000;	  // * scale (4000mu sec) + offset (8000mu sec)
//      double y0 = gaussian(a, b, c, t);
//      double dy = gsl_ran_gaussian (r, 0.1 * y0);
//
//      fit_data.t[i] = t;
//      fit_data.y[i] = y0 + dy;
//    }


    gsl_histogram * h = gsl_histogram_alloc (n);
    gsl_histogram_set_ranges_uniform (h, 0.009000, 0.009000 + 0.002000);

  /*
   * generate data for fitting test
   */
	struct data fit_data;
	gsl_rng * r;
	size_t i;

	gsl_rng_env_setup ();
	r = gsl_rng_alloc (T);

	/* generate synthetic data with noise */
	for (i = 0; i < n; ++i)
	  {
		/* Set range 10ms +- 2ms */
		double t = ((double)i / (double) n) // = (0..1)
			  * 0.002000 + 0.009000;	  // * scale (4000mu sec) + offset (8000mu sec)
		double y0 = gaussian(a, b, c, t);
		double dy = gsl_ran_gaussian (r, 0.1 * y0);

        //gsl_histogram_increment (h, x);
		gsl_histogram_accumulate(h, t, y0 + dy);

	  }

	fit_data.t = h->range;
	fit_data.y = h->bin;
    fit_data.n = n;

  /*
   * 	Starting from here, fitting method setup, TRS
   */

  /* define function parameters to be minimized */
  fdf.f = func_f;			// fitting test to Gaussian
  fdf.df = func_df;			// first derivative Gaussian
  fdf.fvv = func_fvv;		// acceleration method function for Gaussian
  fdf.n = n;				// number of functions => fn(tn) = yn
  fdf.p = p;				// number of independent variables in model
  fdf.params = &fit_data;	// data-vector for the n functions

  /* (Gaussian) fitting model starting parameters */
  gsl_vector_set(x, 0, 8.0);  		/* amplitude */
  gsl_vector_set(x, 1, 0.010200); 	/* center */
  gsl_vector_set(x, 2, 0.001000); 	/* width */

  // enable Levenberg-Marquardt Geodesic acceleration method for trust-region subproblem
  fdf_params.trs = gsl_multifit_nlinear_trs_lmaccel;

  /*
   * Call solver
   */

  int ret;
  struct timespec now, old;
  {

	// get clock, use it as a future reference for update time TIMER_ABS*
	ret = clock_gettime(CLOCK_MONOTONIC, &old);
	if (0 != ret) {
		if (EINTR != ret)
			printf("clock_gettime() failed: %s", strerror(errno));
	}
  }
  solve_system(x, &fdf, &fdf_params);

  {
	ret = clock_gettime(CLOCK_MONOTONIC, &now);
	if (0 != ret) {
		if (EINTR != ret)
			printf("clock_gettime() failed: %s", strerror(errno));
	}

	now.tv_sec -= old.tv_sec;
	now.tv_nsec -= old.tv_nsec;
	tsnorm(&now);

	printf("Solve time: %ld.%09ld\n", now.tv_sec, now.tv_nsec);
  }

  /*
   * print resulting data and model
   * - results vs reality data points
   */
  {
    double A = gsl_vector_get(x, 0);
    double B = gsl_vector_get(x, 1);
    double C = gsl_vector_get(x, 2);

    for (i = 0; i < n; ++i)
      {
        double ti = fit_data.t[i];
        double yi = fit_data.y[i];
        double fi = gaussian(A, B, C, ti);

        printf("%f %f %f\n", ti, yi, fi);
      }
  }

  /*
   * Free vectors and range for Gaussian noise (fake data)
   */
  gsl_vector_free(f);
  gsl_vector_free(x);
  gsl_rng_free(r);

	gsl_histogram_free (h);

  return 0;
}
