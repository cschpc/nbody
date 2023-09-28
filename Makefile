CC=CC
LIBS=
LDFLAGS=-fopenmp
CFLAGS=-DNDEBUG -Ofast -DUSE_MYVECTOR

EXE1=nbody

EXE2=nbody_nb

all: $(EXE1) $(EXE2)

$(EXE1): nbody.cpp myvector.hpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) $(LDFLAGS)

$(EXE2): nbody-nonblocking.cpp myvector.hpp
	$(CC) $(CFLAGS) -o $@ $< $(LIBS) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(EXE1) $(EXE2)
