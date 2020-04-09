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


int runstats_solvehist(stat_hist * h, stat_param * x);

#endif /* RUNSTATS_H_ */
