/*
 * runstats.h
 *
 *  Created on: Apr 8, 2020
 *      Author: Florian Hofer
 */

#ifndef RUNSTATS_H_
#define RUNSTATS_H_

#include <gsl/gsl_histogram.h>
#include <gsl/gsl_vector.h>

// types to abstract and export information
typedef gsl_histogram stat_hist;
typedef gsl_vector stat_param;

int runstats_initparam(stat_param ** x);	// init parameter vector
int runstats_inithist(stat_hist ** h);		// init histogram data structure

int runstats_solvehist(stat_hist * h, stat_param * x);
											// fit model (gaussian) to histogram
int runstats_fithist(stat_hist **h);		// fit histogram bins

int runstats_mdlpdf(stat_hist * h, stat_param * x,
		double t, double * p, double * error);

double runstats_gaussian(const double a, const double b,
		const double c, const double t);

#endif /* RUNSTATS_H_ */
