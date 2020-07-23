CXXC = g++
CFLAGS = -Wall -O3 -fPIC

all: ssl
  
ssl: ssl.o
	$(CXXC) -shared -o libssl.so ssl.o
ssl.o: ssl.cpp ssl.h
	$(CXXC) $(CFLAGS) -c ssl.cpp
clean:
	rm -f *~ ssl.o libssl.so
