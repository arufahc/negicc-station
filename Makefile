CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -fsigned-char -I3rd_party/CrSDK/include -I3rd_party
LDFLAGS = -L3rd_party/CrSDK/lib -lCr_Core -Wl,-rpath,'$$ORIGIN' -lraw -llcms2 -lpthread

BIN_OUT = bin_out
TARGET = capture_test
SRC = main.cpp raw_processor.cpp

all: $(BIN_OUT)/$(TARGET)

$(BIN_OUT)/$(TARGET): $(SRC)
	mkdir -p $(BIN_OUT)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN_OUT)/$(TARGET) $(LDFLAGS)
	cp -r 3rd_party/CrSDK/lib/* $(BIN_OUT)/

clean:
	rm -rf $(BIN_OUT)

.PHONY: all clean
