CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -fsigned-char -I3rd_party/CrSDK/include -I3rd_party -Isrc
LDFLAGS = -L3rd_party/CrSDK/lib -lCr_Core -Wl,-rpath,'$$ORIGIN' -lraw -llcms2 -lpthread

BIN_OUT = build
TARGET = capture_test
TIFF_TARGET = cpp_test_tiff
SRC = src/main.cpp src/raw_processor.cpp src/sony_camera_session.cpp src/image_capture.cpp
TIFF_SRC = src/cpp_test_tiff.cpp src/image_capture.cpp src/raw_processor.cpp src/sony_camera_session.cpp

# Detect nvcc CUDA compiler
NVCC := $(shell which nvcc 2>/dev/null)
CUDA_OBJ :=

ifneq ($(NVCC),)
    CUDA_INC = -I/usr/local/cuda/include -I/usr/local/cuda-12.6/include
    CUDA_LIBS = -L/usr/local/cuda/lib64 -L/usr/local/cuda/targets/aarch64-linux/lib -L/usr/local/cuda-12.6/targets/aarch64-linux/lib -lcudart
    CUDA_OBJ = $(BIN_OUT)/color_conversion_cuda.o
    CXXFLAGS += -DHAVE_CUDA=1 $(CUDA_INC)
    LDFLAGS += $(CUDA_LIBS)
endif

all: $(CUDA_OBJ) $(BIN_OUT)/$(TARGET) $(BIN_OUT)/$(TIFF_TARGET) python_lib

$(BIN_OUT)/color_conversion_cuda.o: src/color_conversion.cu src/color_conversion.h
	mkdir -p $(BIN_OUT)
	nvcc -c src/color_conversion.cu -o $(BIN_OUT)/color_conversion_cuda.o -O3 -std=c++17 -Xcompiler -fPIC

$(BIN_OUT)/$(TARGET): $(SRC) $(CUDA_OBJ)
	mkdir -p $(BIN_OUT)
	$(CXX) $(CXXFLAGS) $(SRC) $(CUDA_OBJ) -o $(BIN_OUT)/$(TARGET) $(LDFLAGS)
	cp -r 3rd_party/CrSDK/lib/* $(BIN_OUT)/

$(BIN_OUT)/$(TIFF_TARGET): $(TIFF_SRC) $(CUDA_OBJ)
	mkdir -p $(BIN_OUT)
	$(CXX) $(CXXFLAGS) $(TIFF_SRC) $(CUDA_OBJ) -o $(BIN_OUT)/$(TIFF_TARGET) $(LDFLAGS)

python_lib:
	if [ ! -d "venv" ]; then python3 -m venv venv && ./venv/bin/pip install --upgrade pip; fi
	./venv/bin/pip install -r requirements.txt setuptools wheel
	./venv/bin/pip install --no-build-isolation .

test_parity: all
	./venv/bin/python3 tests/test_cpython.py
	./venv/bin/python3 tests/test_crosstalk_parity.py
	./venv/bin/python3 tests/test_pipeline_parity.py

test_live: all
	./venv/bin/python3 tests/test_live_parity.py

profile_gen_dry_run: all
	./venv/bin/python3 src/sample_build_prof.py --profile "profiles/profile_Portra400_20260623_170610.json" --reference "http://www.colorreference.de/targets/R190808.zip" --dry-run

profile_gen_dry_run_graph: all
	./venv/bin/python3 src/sample_build_prof.py --profile "profiles/profile_Portra400_20260623_170610.json" --reference "http://www.colorreference.de/targets/R190808.zip" --dry-run --debug

profile_gen_and_convert: all
	@if [ ! -f "sample.ARW" ] && [ -f "test_imgs/sample_portra400.ARW.xz" ]; then \
		echo "Decompressing reference sample from test_imgs..."; \
		xz -d -c test_imgs/sample_portra400.ARW.xz > sample.ARW; \
	elif [ -f "sample.ARW" ] && [ ! -f "test_imgs/sample_portra400.ARW.xz" ]; then \
		echo "Creating compressed copy in test_imgs..."; \
		mkdir -p test_imgs; \
		cp sample.ARW test_imgs/sample_portra400.ARW && xz -z -f test_imgs/sample_portra400.ARW; \
	fi
	./venv/bin/python3 src/sample_build_and_convert.py --profile "profiles/profile_Portra400_20260623_170610.json" --reference "http://www.colorreference.de/targets/R190808.zip" --raw "sample.ARW" --output "build/sample_converted.tiff"

compare_pipelines: all
	@if [ ! -f "sample.ARW" ] && [ -f "test_imgs/sample_portra400.ARW.xz" ]; then \
		echo "Decompressing reference sample from test_imgs..."; \
		xz -d -c test_imgs/sample_portra400.ARW.xz > sample.ARW; \
	elif [ -f "sample.ARW" ] && [ ! -f "test_imgs/sample_portra400.ARW.xz" ]; then \
		echo "Creating compressed copy in test_imgs..."; \
		mkdir -p test_imgs; \
		cp sample.ARW test_imgs/sample_portra400.ARW && xz -z -f test_imgs/sample_portra400.ARW; \
	fi
	./venv/bin/python3 src/sample_build_and_convert.py --profile "profiles/profile_Portra400_20260623_170610.json" --reference "http://www.colorreference.de/targets/R190808.zip" --raw "sample.ARW" --output "build/sample_converted.tiff" --compare

benchmark_cache: all
	@if [ ! -f "sample.ARW" ] && [ -f "test_imgs/sample_portra400.ARW.xz" ]; then \
		echo "Decompressing reference sample from test_imgs..."; \
		xz -d -c test_imgs/sample_portra400.ARW.xz > sample.ARW; \
	fi
	./venv/bin/python3 src/sample_cuda_bench.py

clean:
	rm -rf $(BIN_OUT) negicc_station.egg-info src/color_conversion_cuda.o
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	if [ -d "venv" ]; then ./venv/bin/pip uninstall -y negicc_station || true; fi

.PHONY: all clean python_lib test_parity test_live profile_gen_dry_run profile_gen_dry_run_graph profile_gen_and_convert compare_pipelines benchmark_cache
