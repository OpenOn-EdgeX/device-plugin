CUDA_HOME ?= /usr/local/cuda
CXXFLAGS  := -fPIC -O2 -std=c++17 -I$(CUDA_HOME)/include
LDFLAGS   := -shared -L$(CUDA_HOME)/lib64 -lcuda -lcudart -ldl -lpthread

all: libvai_accelerator.so

libvai_accelerator.so: vai_accelerator.cpp context_manager.hpp routing.hpp
	$(CXX) $(CXXFLAGS) -o $@ vai_accelerator.cpp $(LDFLAGS)

clean:
	rm -f libvai_accelerator.so
