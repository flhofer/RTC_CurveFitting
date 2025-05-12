CC?=$(CROSS_COMPILE)gcc

LIBS	= -lm -lgsl -lgslcblas
# for tests
TLIBS	= -lcheck -lm -lsubunit $(LIBS)

CFLAGS ?= -O2 -Wall -Wno-nonnull -D _GNU_SOURCE
LDFLAGS ?= -L $(OBJDIR) -pthread

.PHONY: all
all: test check

test:
	$(CC) $(CFLAGS) $(LDFLAGS) -o check_test $(wildcard *.c) $(TLIBS)
	
check:
	./check_test

