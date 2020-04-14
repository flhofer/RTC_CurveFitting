#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_histogram.h>

#include <time.h>			// constants and functions for clock
#include <errno.h>			// system error management (LIBC)
#include <string.h>			// strerror print
#include <check.h>			// check C testing library

#include "runstats.h"

#define USEC_PER_SEC		1000000
#define NSEC_PER_SEC		1000000000

struct data
{
	double *t;
	double *y;
	size_t n;
};

/*
 * tsnorm(): verifies timespec for boundaries + fixes it
 *
 * Arguments: pointer to timespec to check
 *
 * Return value: -
 */
static inline void tsnorm(struct timespec *ts)
{
	while (ts->tv_nsec >= NSEC_PER_SEC) {
		ts->tv_nsec -= NSEC_PER_SEC;
		ts->tv_sec++;
	}
}

#define NOITER 5 // number of sampling iterations for fitting

stat_hist * h;
stat_param * x;

void  print_histogram(stat_hist * h, stat_param * x){

	fflush(stdout);
	fflush(stderr);
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

	for (size_t i = 0; i < fit_data.n; ++i)
	  {
		double ti = fit_data.t[i];
		double yi = fit_data.y[i];
		double fi = runstats_gaussian(A, B, C, ti);

		printf("%f %f %f\n", ti, yi, fi);
	  }
	}
}


void test_setup (){
	(void)runstats_inithist(&h);
	(void)runstats_initparam(&x);
}

void test_teardown(){

	print_histogram(h,x);

	/*
	* Free parameter vector and histogram structure
	*/
	gsl_vector_free(x);
	gsl_histogram_free (h);
}

/*
 * generation of random data -> Gaussian with noise
 */
// overshadowing h
size_t generate_histogram(stat_hist * h){
	size_t n = gsl_histogram_bins(h);
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
			double y0 = runstats_gaussian(a, b, c, t);
			double dy = gsl_ran_gaussian (r, 0.1 * y0);

			// insert into histogram
			gsl_histogram_accumulate(h, t, y0 + dy);
		  }
		gsl_rng_free(r);
	}
	return n;
}

/*
 *  Main test program for non-linear weighted least square fitting
 *  initial source taken from https://www.gnu.org/software/gsl/doc/html/nls.html#weighted-nonlinear-least-squares
 */
START_TEST(fitting_check_random)
{
	// do fitting, all except first
	if (_i)
		(void)runstats_fithist(&h);

	(void)printf("***** Solver Iteration %d *****\n", _i+1);
	fflush(stdout);

	(void)generate_histogram(h);

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
	// Max 5..4..3..2..1 ms, the closer we get
	ck_assert_int_le(now.tv_nsec, (NOITER - _i) * 1000000 );
}
END_TEST

START_TEST(fitting_check_probability)
{
	double p = 0, error = 0;
	(void)runstats_mdlpdf(h, x, 0.010450, &p, &error);
	fflush(stdout);

	ck_assert(p >= 0.5);
	ck_assert(error < 0.000005);

}
END_TEST

/*
 * Fitting test suite
 */
void test_fitting (Suite * s) {

	TCase *tc1 = tcase_create("Fitting_random");

	tcase_add_unchecked_fixture(tc1, test_setup, test_teardown);
	tcase_add_loop_test(tc1, fitting_check_random, 0, NOITER);
	tcase_add_test(tc1, fitting_check_probability);

    suite_add_tcase(s, tc1);

	return;
}

/*
 * Setup check runners and return values
 */
int main(void)
{
	// init pseudo-random tables
	srand(time(NULL));

    int nf=0;
    SRunner *sr;

    Suite *s1 = suite_create("Fitting");
    test_fitting(s1);
	sr = srunner_create(s1);
	// No fork needed to keep shared memory across tests
	srunner_set_fork_status (sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    nf += srunner_ntests_failed(sr);
    srunner_free(sr);

    return nf == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
