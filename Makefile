CC := nvcc
CXX := g++

CFLAGS := -std=c++11 --use_fast_math -Xcompiler -fopenmp -Xptxas -O3 -arch=sm_50 -maxrregcount=32
CXXFLAGS := -std=c++11 -O3
LIBS := -lstdc++fs -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lcufft -lgomp -ljetson-inference -ljetson-utils
LDFLAGS := -L/usr/lib/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu/tegra

INCLUDES := -I/usr/include/opencv4 -I/usr/local/include/jetson-inference -I/usr/local/include/jetson-utils -Iincludes

SRC_DIR := src
BUILD_DIR := build

PHYCV_CUDA_DIR := $(SRC_DIR)/phycv_cuda

PHYCV_CUDA_SRCS := $(wildcard $(PHYCV_CUDA_DIR)/*.cu) $(wildcard $(PHYCV_CUDA_DIR)/*.cpp)

PHYCV_CUDA_OBJS := $(patsubst $(PHYCV_CUDA_DIR)/%.cu,$(BUILD_DIR)/%.o,$(PHYCV_CUDA_SRCS))

OBJS := $(PHYCV_CUDA_OBJS) $(PTZ_CAMERA_OBJS)
TARGET := vevid

all: $(BUILD_DIR) $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LIBS)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/%.o: $(PHYCV_CUDA_DIR)/%.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(PHYCV_CUDA_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET)

.PHONY: clean