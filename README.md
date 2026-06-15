# Negative Film Scanning Station (negicc-station)

This repository contains the software for a negative film scanning station designed to run on the **Nvidia Jetson Nano** (ARMv8 64-bit architecture) and interface with a connected **Sony A7R4** camera.

---

## 1. Jetson Nano System Dependencies

Before building and running the scanning software, ensure that the Jetson Nano system is updated and the following system dependencies are installed:

```bash
# Update package list
sudo apt-get update

# Install build tools and C++ compiler
sudo apt-get install -y build-essential g++

# Install SDK and image processing dependencies (USB, XML, LibRaw, and LCMS2)
sudo apt-get install -y libusb-1.0-0-dev libxml2-dev libraw-dev liblcms2-dev
```

Additionally, to allow the application to communicate with the Sony camera over USB without requiring superuser (`root`) privileges, configure the USB udev rules as described in the SDK setup guide.

---

## 2. Sony Camera Remote SDK (CrSDK) Integration

The camera remote control capability relies on the proprietary Sony Camera Remote SDK. Because the SDK is proprietary, its headers and libraries are not stored in this repository.

Please follow the detailed setup instructions in **[3rd_party/CrSDK/README.md](file:///home/alpha/Projects/negicc-station/3rd_party/CrSDK/README.md)** to download, extract, and install the Linux ARMv8 SDK.

---

## 3. Build and Link Configuration (Makefile Flags)

The project includes a **[Makefile](file:///home/alpha/Projects/negicc-station/Makefile)** configured with specific compilation and linking flags optimized for the Jetson Nano (ARM64 architecture) and our library dependencies:

### Compilation Flags (`CXXFLAGS`)
* `-fsigned-char`: **Critical for ARM64 architecture.** By default, `char` is unsigned on ARM64 platforms (unlike x86_64 where it is signed). Since many third-party libraries (including LibRaw headers) expect `char` to be signed, this flag forces `char` to be signed, preventing compilation errors and subtle image parsing bugs.
* `-I3rd_party/CrSDK/include`: Includes the Sony Camera Remote SDK headers.
* `-I3rd_party`: Includes our local third-party headers (such as `lcms2.h` or custom headers).

### Linking Flags (`LDFLAGS`)
* `-L3rd_party/CrSDK/lib -lCr_Core`: Links against the core Sony SDK library.
* `-Wl,-rpath,'$$ORIGIN/3rd_party/CrSDK/lib'`: Sets the run-time shared library search path (rpath) relative to the executable's directory. This allows the application to find `libCr_Core.so` and its adapters at runtime without needing to modify the `LD_LIBRARY_PATH` environment variable.
* `-lraw -llcms2`: Directs the linker to link against `libraw` and `lcms2` system libraries.

> [!IMPORTANT]
> The development packages (`libraw-dev` and `liblcms2-dev`) must be installed on the system beforehand for compilation to succeed. If compilation fails with linker errors like `cannot find -lraw` or `cannot find -llcms2`, make sure you have run:
> ```bash
> sudo apt-get update && sudo apt-get install -y libraw-dev liblcms2-dev
> ```

---

## 4. Building the Project

Once the system dependencies are installed and the Sony SDK files are populated in `3rd_party/CrSDK/`, you can compile the test capture utility:

```bash
# Build the test executable
make

# Run the capture test program
./capture_test
```

---

## 5. Agent Instructions for Managing Dependencies

When introducing any new third-party dependency, library, or system package to this codebase, the agent **MUST** follow these protocol steps to keep the environment reproducible:

1. **Update System Dependencies**: Add any new system-level package requirements to the **Jetson Nano System Dependencies** section in this top-level [README.md](file:///home/alpha/Projects/negicc-station/README.md).
2. **Setup Subdirectory Integration**: If the dependency is a third-party library, create a dedicated folder under `3rd_party/<DependencyName>/` and write a local `README.md` detailing how to download, compile, or install the library.
3. **Configure Git Exclusion**: If the dependency contains proprietary binaries or large compiled libraries, add them to the top-level [.gitignore](file:///home/alpha/Projects/negicc-station/.gitignore) to prevent them from being checked into version control.
4. **Document Code & Builds**: Ensure all Makefiles and source files are updated and linked correctly, and document the complete build instructions so another agent can repeat the execution flow on a fresh Jetson Nano environment.
