#include <stdlib.h>
#include <stdio.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>
#include <gsl/gsl_histogram.h>
#include <gsl/gsl_math.h>

#include <time.h>			// constants and functions for clock
#include <errno.h>			// system error management (LIBC)
#include <string.h>			// strerror print
#include <check.h>			// check C testing library

#include "runstats.h"

/* ----------- EXTERNALLY VISIBLE VARIABLES ----------- */
FILE * dbg_out; // output file stream

/* ----------- LOCALLY VISIBLE VARIABLES ----------- */
static stat_hist * h;
static stat_param * x;
// temporary value to compare last run for random and adapt
static uint32_t nsecPrev;

/* ----------- LOCAL DEFINITIONS ----------- */
#define USEC_PER_SEC		1000000
#define NSEC_PER_SEC		1000000000
#define NOITER 5 // number of sampling iterations for fitting

/* ----------- HELPER FUNCTIONS ----------- */
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

/*
 * verify_histogram () : "verifies" by print or value comparison
 *
 * Arguments: - histogram structure
 * 			  - model parameter vector
 * 			  - print, 0 = check 1% max deviation from peak (A * e) - euler number
 * 			  		   1 = print on screen only
 *
 * Return value: value
 */
static void
histogram_verify(stat_hist * h, stat_param * x, int print){

	/*
	* print resulting data and model
	* - results vs reality data points
	*/
	{
	struct stat_data fit_data = {
			h->range,
			h->bin,
			h->n};

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
			ck_assert( abs(yi-fi)/(A * M_E) <= 0.01 ); // 1% maximum error
	  }
	}
	fflush(stdout);
	fflush(dbg_out);
}

/*
 * histogram_generate() : generation of random data -> Gaussian with noise
 *
 * Arguments: - pointer to histogram structure
 * 			  - amplitude  = number of occurences
 * 			  -	center
 * 			  -	width
 *
 * Return value: - number of generated bins
 */
static size_t
histogram_generate(stat_hist * h, double a, double b, double c, double rand){
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

/*
 * fitting_check() : call curve fitting solver
 *
 * Arguments: - histogram structure
 * 			  - model parameter vector
 *
 * Return value: - elapsed time in nanoseconds (1 second max)
 */
static uint32_t
fitting_check(stat_hist * h, stat_param * x){
	// get time stamp
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

	// update time stamp
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

/*
 * Random fitting tests with high variability
 * nonetheless expect time reduction at every iteration
 * 5..1ms max per cycle
 */
START_TEST(fitting_check_random)
{

	(void)printf("***** Solver Iteration %d *****\n", _i+1);
	fflush(stdout);

	// do fitting, all except first
	if (_i)
		(void)runstats_fithist(&h);
	else // first iteration, set value 10 ms
		nsecPrev = 10000000; // 10 ms start point

	(void)histogram_generate(h, 100, 0.010500, 0.000150, 0.1);

	uint32_t nsec = fitting_check(h, x);

	// Max 5..4..3..2..1 ms, the closer we get
	ck_assert_int_le(nsec, (NOITER - _i) * 1000000 );
	ck_assert_int_lt(nsec, nsecPrev);
	nsecPrev = nsec;
}
END_TEST

/*
 * Adaptive fitting test. Adapting bin number to size
 * every time requiring less time for adaptation
 */
START_TEST(fitting_check_adapt)
{
	(void)printf("***** Solver Iteration %d *****\n", _i+1);

	// do fitting, all except first
	if (_i)
		(void)runstats_fithist(&h);
	else // first iteration, set value 10 ms
		nsecPrev = 10000000; // 10 ms start point

	size_t n = histogram_generate(h, 100, 0.010500, 0.000150,  0.01 / M_E ); // randomize 1%

	(void)printf("Number of bins: %lu\n", n);
	fflush(stdout);

	// run test
	uint32_t nsec = fitting_check(h, x);

	histogram_verify(h,x,0);

	// reduce time needed the closer we get
	ck_assert_int_lt(nsec, nsecPrev);
	nsecPrev = nsec;
}
END_TEST

/*
 * Check equal probability distribution
 * Integrate over resulting parameter vector and
 * verify upper half and lower half contain 50% of probability
 */
START_TEST(fitting_check_probability)
{
	double p = 0, error = 0;
	(void)runstats_mdlpdf(x, 0.010450, gsl_histogram_max(h),&p, &error);
	fflush(stdout);

	ck_assert(p >= 0.5);
	ck_assert(error < 0.000005);

}
END_TEST

/*
 * Merged probability check
 * Curve mix check, adaptation and fix, check 50% fit
 */
START_TEST(fitting_check_merge)
{
	struct {
		double exp;
		double a;
		double b;
		double c;
		} static const merge_bells[5] = {
			{0.010000, 102, 0.010500, 0.000150},
			{0.001000, 60, 0.001020, 0.000090},
			{0.003000,510, 0.003040, 0.000140},
			{0.000200,154, 0.000220, 0.000030},
			{0.010000,324, 0.010500, 0.000090}
			};

	static double merge_avg = 0;

	// local parameters, for merge
	stat_hist * h0;
	stat_param * x0;

	(void)runstats_inithist (&h0, merge_bells[_i].exp);
	(void)runstats_initparam(&x0, merge_bells[_i].exp);

	(void)histogram_generate(h0, merge_bells[_i].a, merge_bells[_i].b, merge_bells[_i].c, 0.009); // randomize 0.9%

	(void)fitting_check(h0, x0);
	histogram_verify(h0,x0, 1);

	(void)gsl_vector_add(x, x0); // add two parameter vectors (central limit theorem)

	merge_avg += gsl_histogram_mean(h0);

	double p = 0, error = 0;
	(void)runstats_mdlpdf(x, merge_avg, 1.0, &p, &error);

	ck_assert(p >= 0.5);
	ck_assert(error < 0.000005);

	(void)gsl_histogram_free(h0);
	(void)gsl_vector_free(x0);
}
END_TEST

/*
 * test_fitting_setup(): init values
 *
 * Arguments: -
 *
 * Return value: -
 */
static void
test_fitting_setup (){
	(void)runstats_inithist (&h, 0.010000);
	(void)runstats_initparam(&x, 0.010000);
}

/*
 * test_fitting_teardown(): clear memory
 *
 * Arguments: -
 *
 * Return value: -
 */
static void
test_fitting_teardown(){

	histogram_verify(h,x, 1);

	/*
	* Free parameter vector and histogram structure
	*/
	gsl_vector_free(x);
	gsl_histogram_free (h);
}

/*
 * test_merge_setup(): init values
 *
 * Arguments: -
 *
 * Return value: -
 */
static void
test_merge_setup (){
	(void)runstats_initparam(&x, 0.010000);
}

/*
 * test_fitting_teardown(): clear memory
 *
 * Arguments: -
 *
 * Return value: -
 */
static void
test_merge_teardown(){
	/*
	* Free parameter vector and histogram structure
	*/
	gsl_vector_free(x);

}

/*
 * TEST SUITES:
 * Fitting test suite
 * CK_RUN_CASE=Fitting_random
 * CK_RUN_CASE=Fitting_adaptation
 */
static void
test_fitting (Suite * s) {

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
 * TEST SUITES:
 * Merging test suite
 * CK_RUN_CASE=Probability_merge_test
 */
static void
test_merge (Suite * s) {

	TCase *tc1 = tcase_create("Probability_merge_test");

	tcase_add_unchecked_fixture(tc1, test_merge_setup, test_merge_teardown);
	tcase_add_loop_test(tc1, fitting_check_merge, 0, NOITER);

    suite_add_tcase(s, tc1);

	return;
}

/*
 * MAIN PROGRAM TEST SUITES:
 * Setup check runners and return values
 */
int
main(void)
{
	// initialize pseudo-random tables
	srand(time(NULL));
	// set debug pipe
	dbg_out = stderr;

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

	fflush(dbg_out);
	fflush(stdout);

    return nf == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
