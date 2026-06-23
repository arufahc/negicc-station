CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -fsigned-char -I3rd_party/CrSDK/include -I3rd_party -Isrc
LDFLAGS = -L3rd_party/CrSDK/lib -lCr_Core -Wl,-rpath,'$$ORIGIN' -lraw -llcms2 -lpthread

BIN_OUT = build
TARGET = capture_test
TIFF_TARGET = cpp_test_tiff
SRC = src/main.cpp src/raw_processor.cpp src/sony_camera_session.cpp src/image_capture.cpp
TIFF_SRC = src/cpp_test_tiff.cpp src/image_capture.cpp src/raw_processor.cpp src/sony_camera_session.cpp

all: $(BIN_OUT)/$(TARGET) $(BIN_OUT)/$(TIFF_TARGET) python_lib

$(BIN_OUT)/$(TARGET): $(SRC)
	mkdir -p $(BIN_OUT)
	$(CXX) $(CXXFLAGS) $(SRC) -o $(BIN_OUT)/$(TARGET) $(LDFLAGS)
	cp -r 3rd_party/CrSDK/lib/* $(BIN_OUT)/

$(BIN_OUT)/$(TIFF_TARGET): $(TIFF_SRC)
	mkdir -p $(BIN_OUT)
	$(CXX) $(CXXFLAGS) $(TIFF_SRC) -o $(BIN_OUT)/$(TIFF_TARGET) $(LDFLAGS)

python_lib:
	if [ ! -d "venv" ]; then python3 -m venv venv && ./venv/bin/pip install --upgrade pip; fi
	./venv/bin/pip install -r requirements.txt setuptools wheel
	./venv/bin/pip install --no-build-isolation .

test_parity: all
	./venv/bin/python3 tests/test_cpython.py

test_live: all
	./venv/bin/python3 tests/test_live_parity.py

profile_gen_dry_run: all
	./venv/bin/python3 src/sample_build_prof.py --profile "profiles/profile_Portra 400_20260623_000121.json" --reference "http://www.colorreference.de/targets/R190808.zip" --dry-run

profile_gen_dry_run_graph: all
	./venv/bin/python3 src/sample_build_prof.py --profile "profiles/profile_Portra 400_20260623_000121.json" --reference "http://www.colorreference.de/targets/R190808.zip" --dry-run --debug

clean:
	rm -rf $(BIN_OUT) negicc_station.egg-info
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	if [ -d "venv" ]; then ./venv/bin/pip uninstall -y negicc_station || true; fi


.PHONY: all clean python_lib test_parity test_live profile_gen_dry_run profile_gen_dry_run_graph
