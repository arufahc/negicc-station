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

# Install SDK dependencies (USB library and XML parsing support)
sudo apt-get install -y libusb-1.0-0-dev libxml2-dev
```

Additionally, to allow the application to communicate with the Sony camera over USB without requiring superuser (`root`) privileges, configure the USB udev rules as described in the SDK setup guide.

---

## 2. Sony Camera Remote SDK (CrSDK) Integration

The camera remote control capability relies on the proprietary Sony Camera Remote SDK. Because the SDK is proprietary, its headers and libraries are not stored in this repository.

Please follow the detailed setup instructions in **[3rd_party/CrSDK/README.md](file:///home/alpha/Projects/negicc-station/3rd_party/CrSDK/README.md)** to download, extract, and install the Linux ARMv8 SDK.

---

## 3. Building the Project

Once the system dependencies are installed and the Sony SDK files are populated in `3rd_party/CrSDK/`, you can compile the test capture utility:

```bash
# Build the test executable
make

# Run the capture test program
./capture_test
```

---

## 4. Agent Instructions for Managing Dependencies

When introducing any new third-party dependency, library, or system package to this codebase, the agent **MUST** follow these protocol steps to keep the environment reproducible:

1. **Update System Dependencies**: Add any new system-level package requirements to the **Jetson Nano System Dependencies** section in this top-level [README.md](file:///home/alpha/Projects/negicc-station/README.md).
2. **Setup Subdirectory Integration**: If the dependency is a third-party library, create a dedicated folder under `3rd_party/<DependencyName>/` and write a local `README.md` detailing how to download, compile, or install the library.
3. **Configure Git Exclusion**: If the dependency contains proprietary binaries or large compiled libraries, add them to the top-level [.gitignore](file:///home/alpha/Projects/negicc-station/.gitignore) to prevent them from being checked into version control.
4. **Document Code & Builds**: Ensure all Makefiles and source files are updated and linked correctly, and document the complete build instructions so another agent can repeat the execution flow on a fresh Jetson Nano environment.
