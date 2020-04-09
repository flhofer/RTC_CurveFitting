#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_multifit_nlinear.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_histogram.h>

#include <time.h>			// constants and functions for clock
#include <errno.h>			// system error management (LIBC)
#include <string.h>			// strerror print

#include "runstats.h"

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

#define NOITER 5 // number of sampling iterations for fitting

/*
 *  Main test program for non-linear weighted least square fitting
 *  initial source taken from https://www.gnu.org/software/gsl/doc/html/nls.html#weighted-nonlinear-least-squares
 */

int
main (void)
{
	gsl_histogram * h;
	stat_param * x;

	(void)runstats_inithist(&h);
	(void)runstats_initparam(&x);

	size_t n;

	// Solver iterations start here
	for ( int run_iter=0; run_iter < NOITER; run_iter++) {

		// do fitting, all except first
		if (run_iter)
			(void)runstats_fithist(&h);

		(void)printf("***** Solver Iteration %d *****\n", run_iter+1);
		fflush(stdout);
		n = gsl_histogram_bins(h);
		/*
		 * generation of random data -> Gaussian with noise
		 */
		{
			// init random Gaussian noise
			const gsl_rng_type * T = gsl_rng_default;
			gsl_rng * r;
			gsl_rng_env_setup ();
			r = gsl_rng_alloc (T);

			// synthetic Gaussian parameters
			const double a = 100.0;  	/* amplitude */ // = number of occurences
			const double b = 0.010500;  /* center */
			const double c = 0.000150;  /* width */

			/* generate synthetic data with noise */
			for (size_t i = 0; i < n; ++i)
			  {
				// get range of bin
				double t,t1;
				(void)gsl_histogram_get_range(h, i, &t, &t1);

				// take average value
				t= (t+t1)/2;

				// compute Gaussian value and random noise
				double y0 = gaussian(a, b, c, t);
				double dy = gsl_ran_gaussian (r, 0.1 * y0);

				// insert into histogram
				gsl_histogram_accumulate(h, t, y0 + dy);
			  }
			gsl_rng_free(r);
		}


		// get timestamp
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

		/*
		* Call solver
		*/
		(void)runstats_solvehist(h,x);

		// update timestamp
		{
			ret = clock_gettime(CLOCK_MONOTONIC, &now);
			if (0 != ret) {
				if (EINTR != ret)
					printf("clock_gettime() failed: %s", strerror(errno));
			}

			// compute difference -> time needed
			now.tv_sec -= old.tv_sec;
			now.tv_nsec -= old.tv_nsec;
			tsnorm(&now);

			printf("Solve time: %ld.%09ld\n", now.tv_sec, now.tv_nsec);
		}

	    //gsl_histogram_fprintf (stdout, h, "%3.05f", "%3.05f");

	}
	/*
	* print resulting data and model
	* - results vs reality data points
	*/
	{
	struct data fit_data;
	// pass histogram to fitting structure
	fit_data.t = h->range;
	fit_data.y = h->bin;
	fit_data.n = h->n;


	double A = gsl_vector_get(x, 0);
	double B = gsl_vector_get(x, 1);
	double C = gsl_vector_get(x, 2);

	for (size_t i = 0; i < n; ++i)
	  {
		double ti = fit_data.t[i];
		double yi = fit_data.y[i];
		double fi = gaussian(A, B, C, ti);

		printf("%f %f %f\n", ti, yi, fi);
	  }
	}

	/*
	* Free parameter vector and histogram structure
	*/
	gsl_vector_free(x);
	gsl_histogram_free (h);

	return 0;
}
