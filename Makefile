CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -fsigned-char -I3rd_party/CrSDK/include -I3rd_party
LDFLAGS = -L3rd_party/CrSDK/lib -lCr_Core -Wl,-rpath,'$$ORIGIN/3rd_party/CrSDK/lib' -L3rd_party/libs -lraw -llcms2 -lpthread

TARGET = capture_test
SRC = main.cpp raw_processor.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
