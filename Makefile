CC := nvcc
CXX := g++

CFLAGS := -std=c++11 --use_fast_math -Xcompiler -fopenmp -Xptxas -O3 -arch=sm_50 -maxrregcount=32
CXXFLAGS := -std=c++11 -O3
LIBS := -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui -lopencv_videoio -lcufft -lgomp -ljetson-inference -ljetson-utils
LDFLAGS := -L/usr/lib/aarch64-linux-gnu -L/usr/lib/aarch64-linux-gnu/tegra

INCLUDES := -I/usr/include/opencv4 -I/usr/local/include/jetson-inference -I/usr/local/include/jetson-utils -Iincludes

SRC_DIR := src
BUILD_DIR := build

SRCS := $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(SRCS))
TARGET := vevid

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDES) $(LDFLAGS) $^ -o $@ $(LIBS)

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(BUILD_DIR)/*.o $(TARGET)
