# RTC_CurveFitting
Test code for runtime analysis and curve fitting in C with focus on resource efficiency and speed. It focuses on the application for  periodic deadline scheduled tasks, their behavior and their run-time estimation.

The code and tests in this library are made to fulfill the following steps:

1. collect run-time data of periodic tasks in a histogram
1. solving and curve fitting of the data to a Gaussian (or more) model
1. Automatic adaptation and fitting of the bin-size of the histogram based on numbers and data
1. probability estimation (integration) of a given run-time
1. maximum run-time estimation to stay within a given probability of success to not miss a deadline
   