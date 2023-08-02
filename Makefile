CXX=g++
CUX=/usr/local/cuda/bin/nvcc
    
CFLAGS=-std=c++14 -O3 -Wall -march=native -mavx2 -mfma -mno-avx512f -fopenmp -I/usr/local/cuda/include
CUDA_CFLAGS:=$(foreach option, $(CFLAGS),-Xcompiler=$(option))
LDFLAGS=-pthread -L/usr/local/cuda/lib64
LDLIBS=-lstdc++ -lcudart -lm

TARGET=model
OBJECTS=main.o model.o tensor.o util.o

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS) $(LDLIBS)

%.o: %.cpp
	$(CXX) $(CFLAGS) -c -o $@ $^

%.o: %.cu
	$(CUX) $(CUDA_CFLAGS) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
