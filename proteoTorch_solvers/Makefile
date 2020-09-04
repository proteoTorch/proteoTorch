CXX ?= g++
CC ?= gcc
CFLAGS = -Wall -Wconversion -O3 -fPIC
LIBS = blas/blas.a
OS = $(shell uname)
# LIBS = -lblas

all: ssl
  
ssl: ssl.o blas/blas.a
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libssl.so"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libssl.o"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} ssl.o blas/blas.a -o libssl.so

ssl.o: ssl.cpp ssl.h
	$(CXX) $(CFLAGS) -c -o ssl.o ssl.cpp

blas/blas.a: blas/*.c blas/*.h
	make -C blas OPTFLAGS='$(CFLAGS)' CC='$(CC)';

clean:
	make -C blas clean
	rm -f *~ ssl.o libssl.so
