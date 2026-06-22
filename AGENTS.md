# Agent Instructions for negicc-station

This file contains mandatory rules for any agent working in this repository.

---

## Project Objective

The primary objective of `negicc-station` is to interface with a connected **Sony A7R4** camera over USB on an **Nvidia Jetson Nano** running Linux, trigger live captures (supporting both single-shot and Sony 4-shot IBIS pixel shift), perform high-performance linear RAW image decoding/merging in C++ (handling Bayer grid demosaicing or 4-frame alignment without interpolation to prevent grain bleeding), and expose the resulting linear RGB data to Python as a standard 16-bit NumPy array. This library forms the core capture and processing frontend for a film negative scanning system, enabling downstream mathematical negative inversion.

---

## Documentation Rules

### Never Use Hardcoded Absolute Paths

Do **not** embed machine-specific absolute paths (e.g., `/home/user/Projects/negicc-station/...`) anywhere in documentation, markdown files, scripts, or source comments.

This applies to:
- Markdown links (e.g., `[text](file:///home/user/Projects/...)` is **forbidden**.)
- Shell script examples (e.g., `export LD_LIBRARY_PATH=...:/home/user/Projects/...` is **forbidden**.)

- Any hardcoded path that assumes a specific user or machine layout

**Use relative paths instead:**
- Markdown links: `[text](relative/path/to/file.md)`
- Shell examples that require an absolute path: use `$(pwd)/relative/path` so it resolves correctly on any machine

**Why:** This repo is intended to run on a Jetson Nano (and potentially other machines). Hardcoded paths break portability and are misleading in documentation shared across environments.

---

## Dependency Management Protocol

When introducing any new third-party dependency, library, or system package:

1. **Update System Dependencies** — Add new system-level packages to the **Jetson Nano System Dependencies** section in the top-level [README.md](README.md).
2. **Setup Subdirectory** — For third-party libraries, create a folder under `3rd_party/<DependencyName>/` with a local `README.md` explaining how to download, compile, or install it.
3. **Configure Git Exclusion** — If the dependency includes proprietary binaries or large compiled files, add entries to [.gitignore](.gitignore).
4. **Document Builds** — Update all relevant Makefiles and document complete build instructions so the setup can be reproduced on a fresh Jetson Nano.
5. **Python Dependencies** — Encode any Python-level dependencies in `requirements.txt` at the repository root. Always keep this file up to date whenever a new Python dependency is introduced.

---

## Git Configuration Rules

All agents committing to this repository must configure their Git author information as follows:
- **Name**: `Agent for Alpha Lam`
- **Email**: `arufa.hc@gmail.com`

**Command to run at startup:**
```bash
git config user.name "Agent for Alpha Lam" && git config user.email "arufa.hc@gmail.com"
```

---

## Source Code and Script Placement

All new source files, testing scripts, and sample programs (such as sample UI tools and auto-exposure experiments) must be placed in the `src/` directory. Do not add raw source files or executable python scripts directly to the repository root to keep the root directory clean.


