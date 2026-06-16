# Agent Instructions for negicc-station

This file contains mandatory rules for any agent working in this repository.

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
