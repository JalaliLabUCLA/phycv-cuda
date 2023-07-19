CC := nvcc
CXX := g++

CFLAGS := -std=c++11 -O3 --use_fast_math -Xcompiler -fopenmp -Xptxas -O3 -arch=sm_50 -maxrregcount=32
CXXFLAGS := -std=c++11 -O3
LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lcufft -lgomp
LDFLAGS := -L/usr/lib/aarch64-linux-gnu

INCLUDES := -I/usr/include/opencv4

SRCS := vevid.cu kernels.cu main.cpp video.cpp
OBJS := $(SRCS:.cu=.o)
TARGET := vevid

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LIBS)

%.o: %.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS)
