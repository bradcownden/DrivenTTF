CFLAGS = -Wall -O3 -c
LFLAGS = -lm -lgsl -lm

all:
	gcc $(CFLAGS) NN_equalfreq_integrals.cpp 
	gcc NN_equalfreq_integrals.o $(LFLAGS) -o NN_equalfreq_integrals
