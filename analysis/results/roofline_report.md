# ConvStencil Roofline Analysis Report

## Hardware Specifications

- Peak Compute Performance: 4838 GFLOPS
- Peak Memory Bandwidth: 1555 GB/s
- Ridge Point: 3.11 FLOPS/Byte

## Configuration Analysis

### star_1d1r_(8192,)
- Arithmetic Intensity: 0.250 FLOPS/Byte
- Achieved Performance: 507.00 GFLOPS
- Expected Performance: 388.75 GFLOPS
- Efficiency: 130.4%
- Bottleneck: Memory

### box_2d1r_(512, 512)
- Arithmetic Intensity: 2.233 FLOPS/Byte
- Achieved Performance: 1028.16 GFLOPS
- Expected Performance: 3471.58 GFLOPS
- Efficiency: 29.6%
- Bottleneck: Memory

### star_2d1r_(1024, 1024)
- Arithmetic Intensity: 1.245 FLOPS/Byte
- Achieved Performance: 843.25 GFLOPS
- Expected Performance: 1936.18 GFLOPS
- Efficiency: 43.6%
- Bottleneck: Memory

### box_3d1r_(64, 64, 64)
- Arithmetic Intensity: 6.155 FLOPS/Byte
- Achieved Performance: 1028.16 GFLOPS
- Expected Performance: 4112.64 GFLOPS
- Efficiency: 25.0%
- Bottleneck: Compute

## Bottleneck Summary

- Memory-bound configurations: 3
- Compute-bound configurations: 1

## Optimization Suggestions

### star_1d1r_(8192,)
- Increase arithmetic intensity by fusing operations
- Optimize data layout for better cache utilization
- Use blocking/tiling to improve temporal locality
- Consider data compression techniques

### box_2d1r_(512, 512)
- Increase arithmetic intensity by fusing operations
- Optimize data layout for better cache utilization
- Use blocking/tiling to improve temporal locality
- Consider data compression techniques

### star_2d1r_(1024, 1024)
- Increase arithmetic intensity by fusing operations
- Optimize data layout for better cache utilization
- Use blocking/tiling to improve temporal locality
- Consider data compression techniques

### box_3d1r_(64, 64, 64)
- Optimize tensor core utilization
- Improve stencil2row transformation efficiency
- Use dual tessellation for better parallelism
- Optimize kernel fusion strategies

