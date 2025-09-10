# ConvStencil Modeling Analysis

This directory contains the modeling analysis source code for ConvStencil, providing performance modeling, benchmarking, and analysis tools for tensor core-based stencil computations.

## Components

### Performance Models
- `performance_model.py` - Core performance modeling framework
- `tensor_core_model.py` - Tensor Core specific performance models
- `memory_model.py` - Memory bandwidth and latency models
- `roofline_model.py` - Roofline performance analysis

### Benchmarking Tools
- `benchmark_runner.py` - Automated benchmarking framework
- `comparison_tools.py` - Tools for comparing against cuDNN, AMOS, etc.
- `metrics_collector.py` - Performance metrics collection

### Analysis Scripts
- `performance_analysis.py` - Main performance analysis script
- `scaling_analysis.py` - Scaling behavior analysis
- `efficiency_analysis.py` - Compute and memory efficiency analysis
- `visualization.py` - Results visualization and plotting

## Usage

### Basic Performance Analysis
```bash
python analysis/performance_analysis.py --shape box2d1r --size 1024 1024 --iterations 100
```

### Comprehensive Benchmarking
```bash
python analysis/benchmark_runner.py --config configs/benchmark_config.yaml
```

### Model Validation
```bash
python analysis/model_validation.py --measured_data results/measurements.json
```

## Model Components

The performance model includes:

1. **Tensor Core Utilization Model** - Predicts tensor core efficiency based on workload characteristics
2. **Memory Bandwidth Model** - Models data movement costs and memory access patterns
3. **Stencil2Row Transformation Model** - Analyzes the cost and efficiency of layout transformation
4. **Dual Tessellation Model** - Models the impact of tessellation strategies
5. **Conflict Removal Model** - Analyzes lookup table and dirty bits padding effectiveness

## Dependencies

- NumPy
- Matplotlib
- Pandas
- SciPy
- PyYAML