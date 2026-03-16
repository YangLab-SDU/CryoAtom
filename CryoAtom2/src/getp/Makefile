CXX=g++

CXX_FLAGS=-std=c++11 -fopenmp -O2 -w -lm

#CXX_FLAGS=-std=c++0x -fopenmp -O2 -w -lm -static
#CXX_FLAGS=-std=c++0x -fopenmp -O3 -w -lm -static

TARGET=getp

SRCS=mrcio.cpp ms.cpp parser.cpp grid.cpp pdbio.cpp
 
OBJS = $(SRCS:.cpp=.o)

%.o: %.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

all: $(TARGET)

getp: parser.o grid.o pdbio.o mrcio.o ms.o getp.cpp
	$(CXX) $(CXX_FLAGS) parser.o grid.o pdbio.o mrcio.o ms.o getp.cpp -o getp

clean :
	rm *.o getp

.PHONY: all clean
