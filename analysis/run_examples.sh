#!/bin/bash

# ConvStencil Analysis Examples
# This script demonstrates how to use the modeling analysis tools

echo "ConvStencil Modeling Analysis Examples"
echo "======================================"

# Ensure we're in the analysis directory
cd "$(dirname "$0")"

# Create output directories
mkdir -p results plots

echo ""
echo "1. Basic Performance Analysis"
echo "-----------------------------"
python performance_analysis.py --shape box_2d1r --size 1024 1024 --iterations 100 --mode analyze

echo ""
echo "2. Scaling Analysis"
echo "-------------------"
python performance_analysis.py --shape box_2d1r --size 512 512 --iterations 100 --mode analyze --scaling-analysis --save-plots

echo ""
echo "3. Model Validation (requires built executables)"
echo "------------------------------------------------"
if [ -f "../build/convstencil_2d" ]; then
    python performance_analysis.py --shape box_2d1r --size 256 256 --iterations 10 --mode model_validation --save-plots
else
    echo "Skipping model validation - convstencil_2d not found in ../build/"
fi

echo ""
echo "4. Comprehensive Benchmark Suite"
echo "--------------------------------"
if [ -f "../build/convstencil_1d" ] && [ -f "../build/convstencil_2d" ]; then
    python benchmark_runner.py --config configs/benchmark_config.yaml
else
    echo "Skipping benchmark suite - executables not found in ../build/"
fi

echo ""
echo "5. Tensor Core Analysis Examples"
echo "--------------------------------"
python -c "
from tensor_core_model import TensorCoreModel, TensorCoreConfig
from performance_model import HardwareSpec, StencilConfig, StencilShape

hardware = HardwareSpec()
tensor_config = TensorCoreConfig()
model = TensorCoreModel(hardware, tensor_config)

config = StencilConfig(
    shape=StencilShape.BOX_2D1R,
    dimensions=(1024, 1024),
    iterations=100,
    radius=1,
    halo_size=1
)

result = model.predict_tensor_core_performance(config)
print('Tensor Core Analysis Results:')
for key, value in result.items():
    if isinstance(value, float):
        print(f'  {key}: {value:.4f}')
    else:
        print(f'  {key}: {value}')
"

echo ""
echo "6. Memory Analysis Examples"
echo "---------------------------"
python -c "
from memory_model import MemoryModel
from performance_model import HardwareSpec, StencilConfig, StencilShape

hardware = HardwareSpec()
model = MemoryModel(hardware)

config = StencilConfig(
    shape=StencilShape.STAR_2D1R,
    dimensions=(2048, 2048),
    iterations=100,
    radius=1,
    halo_size=1
)

result = model.predict_memory_performance(config)
print('Memory Analysis Results:')
print(f'  Memory bound time: {result[\"memory_bound_time_seconds\"]:.6f} seconds')
print(f'  Memory efficiency: {result[\"memory_efficiency\"]*100:.2f}%')
print(f'  L1 hit rate: {result[\"cache_behavior\"][\"l1_hit_rate\"]*100:.2f}%')
print(f'  Bandwidth utilization: {result[\"bandwidth_utilization\"][\"bandwidth_utilization\"]*100:.2f}%')
"

echo ""
echo "Analysis examples completed!"
echo "Check the 'results/' directory for output files and 'plots/' for visualizations."