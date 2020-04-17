#ifndef __ERROR_H
	#define __ERROR_H

	#include <stdio.h>
	#include <stdlib.h>
	#include <stdarg.h>
	#include <string.h>

	#if DEBUG
		extern FILE * dbg_out; // debug output file, defined in main
		#define printDbg(...) (void)fprintf (dbg_out, __VA_ARGS__)
	#else
		#define printDbg(...) //
	#endif

	// general log information
	void debug(char *fmt, ...);
	void cont(char *fmt, ...);
	void info(char *fmt, ...);
	void warn(char *fmt, ...);
	// error only printing
	void err_msg(char *fmt, ...);
	void err_msg_n(int err, char *fmt, ...);
	// normal exit on error
	void err_exit(char *fmt, ...) __attribute__((noreturn));
	void err_exit_n(int err, char *fmt, ...) __attribute__((noreturn));
	// fatal errors, immediate exit (abort)
	void fatal(char *fmt, ...) __attribute__((noreturn));
	void fatal_n(int err, char *fmt, ...) __attribute__((noreturn));
	// interal error print function
	void err_doit(int err, const char *fmt, va_list ap);

	/* exit codes */
	#define EXIT_SUCCESS 0
	#define EXIT_FAILURE 1
	#define EXIT_INV_CONFIG 2
	#define EXIT_INV_COMMANDLINE 3

#endif	/* __ERROR_H */
