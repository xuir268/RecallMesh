# Contributing

Use Python 3.9+ and a C++20 compiler. Install the build dependencies and package:

```sh
python -m pip install scikit-build-core nanobind numpy pytest
python -m pip install -e . --no-build-isolation
python -m pytest tests -q
cmake -S . -B build/native
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```

Keep the existing RAM API compatible. Add tests for changes to persistence, quotas, stable IDs, or the JSON protocol. Protocol additions should preserve version-1 requests and config defaults. Do not commit personal memory databases, model weights, or benchmark datasets. Report benchmark methodology and failures alongside improvements.
