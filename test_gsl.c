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

size_t generate_histogram(stat_hist *, double, double, double, double rand);

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

void  verify_histogram(stat_hist * h, stat_param * x, int print){

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

		if (print)
			printf("%f %f %f\n", ti, yi, fi);
		else
			ck_assert( abs(yi-fi) <= (A * 0.01) ); // 1% maximum error
	  }
	}
}

void test_fitting_setup (){
	(void)runstats_inithist (&h, 0.010000);
	(void)runstats_initparam(&x, 0.010000);
}

void test_fitting_teardown(){

	verify_histogram(h,x, 1);

	/*
	* Free parameter vector and histogram structure
	*/
	gsl_vector_free(x);
	gsl_histogram_free (h);
}

/*
 * generate_histogram() : generation of random data -> Gaussian with noise
 *
 * Arguments: - pointer to histogram structure
 * 			  - amplitude  = number of occurences
 * 			  -	center
 * 			  -	width
 *
 * Return value: - number of generated bins
 */
size_t generate_histogram(stat_hist * h, double a, double b, double c, double rand){
	size_t n = gsl_histogram_bins(h);
	{
		// init random Gaussian noise
		const gsl_rng_type * T = gsl_rng_default;
		gsl_rng * r;
		gsl_rng_env_setup ();
		r = gsl_rng_alloc (T);


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
			double dy = gsl_ran_gaussian (r, rand * y0);

			// insert into histogram
			gsl_histogram_accumulate(h, t, y0 + dy);
		  }
		gsl_rng_free(r);
	}
	return n;
}

uint32_t fitting_check(stat_hist * h, stat_param * x){
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
	return now.tv_nsec;
}

START_TEST(fitting_check_random)
{
	(void)printf("***** Solver Iteration %d *****\n", _i+1);
	fflush(stdout);

	// do fitting, all except first
	if (_i)
		(void)runstats_fithist(&h);

	(void)generate_histogram(h, 100, 0.010500, 0.000150, 0.1);

	uint32_t nsec = fitting_check(h, x);

	// Max 5..4..3..2..1 ms, the closer we get
	ck_assert_int_le(nsec, (NOITER - _i) * 1000000 );
}
END_TEST

uint32_t nsecold = 10000000; // 10 ms start point

START_TEST(fitting_check_adapt)
{
	(void)printf("***** Solver Iteration %d *****\n", _i+1);

	// do fitting, all except first
	if (_i)
		(void)runstats_fithist(&h);

	size_t n = generate_histogram(h, 100, 0.010500, 0.000150, 0.009); // randomize 0.9%

	(void)printf("Number of bins: %lu\n", n);
	fflush(stdout);


	uint32_t nsec = fitting_check(h, x);

	verify_histogram(h,x,0);

	// Max 5..4..3..2..1 ms, the closer we get
	ck_assert_int_le(nsec, nsecold);
}
END_TEST

START_TEST(fitting_check_probability)
{
	double p = 0, error = 0;
	(void)runstats_mdlpdf(x, 0.010450, gsl_histogram_max(h),&p, &error);
	fflush(stdout);

	ck_assert(p >= 0.5);
	ck_assert(error < 0.000005);

}
END_TEST

struct {
	double exp;
	double a;
	double b;
	double c;
} merge_bells[5] = {
	{0.010000, 102, 0.010500, 0.000150},
	{0.001000, 60, 0.001020, 0.000090},
	{0.003000,510, 0.003040, 0.000140},
	{0.000200,154, 0.000220, 0.000030},
	{0.010000,324, 0.010500, 0.000090}
};

double merge_avg = 0;

START_TEST(fitting_check_merge)
{

	stat_hist * h0;
	stat_param * x0;
	(void)runstats_inithist (&h0, merge_bells[_i].exp);
	(void)runstats_initparam(&x0, merge_bells[_i].exp);

	(void)generate_histogram(h0, merge_bells[_i].a, merge_bells[_i].b, merge_bells[_i].c, 0.009); // randomize 0.9%

	(void)fitting_check(h0, x0);
	verify_histogram(h0,x0, 1);

	(void)gsl_vector_add(x, x0); // add two parameter vectors (central limit theorem)

	merge_avg += gsl_histogram_mean(h0);

	double p = 0, error = 0;
	(void)runstats_mdlpdf(x, merge_avg, 1.0, &p, &error);
	fflush(stdout);

	ck_assert(p >= 0.5);
	ck_assert(error < 0.000005);

	(void)gsl_histogram_free(h0);
	(void)gsl_vector_free(x0);
}
END_TEST


/*
 * Fitting test suite
 */
void test_fitting (Suite * s) {

	TCase *tc1 = tcase_create("Fitting_random");

	tcase_add_unchecked_fixture(tc1, test_fitting_setup, test_fitting_teardown);
	tcase_add_loop_test(tc1, fitting_check_random, 0, NOITER);
	tcase_add_test(tc1, fitting_check_probability);

    suite_add_tcase(s, tc1);

	TCase *tc2 = tcase_create("Fitting_adaptation");

	tcase_add_unchecked_fixture(tc2, test_fitting_setup, test_fitting_teardown);
	tcase_add_loop_test(tc2, fitting_check_adapt, 0, NOITER);
	tcase_add_test(tc2, fitting_check_probability);

    suite_add_tcase(s, tc2);

	return;
}

/*
 * Merging test suite
 */
void test_merge (Suite * s) {

	TCase *tc1 = tcase_create("Probability_merge_test");

	// TODO: Teardown print does not work yet -> merge is way higher
	tcase_add_unchecked_fixture(tc1, test_fitting_setup, test_fitting_teardown);
	tcase_add_loop_test(tc1, fitting_check_merge, 0, NOITER);

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
    test_merge(s1);
    sr = srunner_create(s1);
	// No fork needed to keep shared memory across tests
	srunner_set_fork_status (sr, CK_NOFORK);
    srunner_run_all(sr, CK_NORMAL);
    nf += srunner_ntests_failed(sr);
    srunner_free(sr);

	fflush(stdout);
	fflush(stderr);

    return nf == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
