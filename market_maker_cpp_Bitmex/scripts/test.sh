#!/bin/bash
set -e

# Navigate to build directory
cd build

# Run tests
ctest --output-on-failure -j$(nproc)

# Run memory checks if requested
if [ "$1" == "--memcheck" ]; then
    echo "Running memory checks..."
    ctest -T memcheck
fi

# Generate coverage report if requested
if [ "$1" == "--coverage" ]; then
    echo "Generating coverage report..."
    lcov --capture --directory . --output-file coverage.info
    genhtml coverage.info --output-directory coverage_report
fi

echo "All tests completed successfully!" 