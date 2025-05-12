CC?=$(CROSS_COMPILE)gcc

LIBS	= -lm -lgsl -lgslcblas
# for tests
TLIBS	= -lcheck -lsubunit $(LIBS)

CFLAGS ?= -O0 -D DEBUG -Wall -Wno-nonnull -D _GNU_SOURCE

.PHONY: all
all: test check

test:
	$(CC) $(CFLAGS) -o check_test $(wildcard *.c) $(TLIBS)
	
check:
	./check_test

