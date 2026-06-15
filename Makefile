CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -I3rd_party/CrSDK/include
LDFLAGS = -L3rd_party/CrSDK/lib -lCr_Core -Wl,-rpath,'$$ORIGIN/3rd_party/CrSDK/lib' -lpthread

TARGET = capture_test
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
