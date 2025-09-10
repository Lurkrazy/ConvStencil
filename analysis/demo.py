#!/usr/bin/env python3
"""
ConvStencil Modeling Analysis Demo

This script demonstrates the modeling analysis capabilities for ConvStencil.
It shows how to use the performance models, analyze different configurations,
and generate visualizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from performance_model import PerformanceModel, StencilConfig, StencilShape, HardwareSpec, TensorCoreConfig
from tensor_core_model import TensorCoreModel
from memory_model import MemoryModel
from roofline_model import RooflineModel, RooflinePoint
from visualization import PerformanceVisualizer

def demo_basic_performance_analysis():
    """Demonstrate basic performance analysis"""
    print("=== Basic Performance Analysis Demo ===")
    
    # Setup hardware and configuration
    hardware = HardwareSpec()
    tensor_config = TensorCoreConfig()
    
    # Create a 2D box stencil configuration
    config = StencilConfig(
        shape=StencilShape.BOX_2D1R,
        dimensions=(1024, 1024),
        iterations=100,
        radius=1,
        halo_size=1
    )
    
    # Initialize performance model
    perf_model = PerformanceModel(hardware, tensor_config)
    
    # Predict performance
    results = perf_model.predict_performance(config)
    
    print(f"Configuration: {config.shape.value} {config.dimensions}")
    print(f"Predicted execution time: {results['predicted_time_seconds']:.6f} seconds")
    print(f"Throughput: {results['achieved_flops_per_second']/1e9:.2f} GFLOPS")
    print(f"Efficiency: {results['efficiency_percent']:.2f}%")
    print(f"Tensor Core Utilization: {results['tensor_core_utilization']*100:.2f}%")
    print(f"Arithmetic Intensity: {results['arithmetic_intensity']:.3f} FLOPS/Byte")
    print(f"Bottleneck: {results['bottleneck']}")
    print()
    
    return results

def demo_tensor_core_analysis():
    """Demonstrate tensor core specific analysis"""
    print("=== Tensor Core Analysis Demo ===")
    
    hardware = HardwareSpec()
    tensor_config = TensorCoreConfig()
    tensor_model = TensorCoreModel(hardware, tensor_config)
    
    config = StencilConfig(
        shape=StencilShape.STAR_2D1R,
        dimensions=(2048, 2048),
        iterations=50,
        radius=1,
        halo_size=1
    )
    
    # Analyze tensor core performance
    results = tensor_model.predict_tensor_core_performance(config)
    
    print(f"Configuration: {config.shape.value} {config.dimensions}")
    print(f"Tensor Core Efficiency: {results['tensor_core_efficiency']*100:.2f}%")
    print(f"Achieved Performance: {results['achieved_tflops']:.3f} TFLOPS")
    print(f"Tensor Core Utilization: {results['tensor_core_utilization']*100:.2f}%")
    print(f"Operations Required: {results['tensor_core_ops_required']:,}")
    print()
    
    # Analyze different tessellation strategies
    tessellation_factors = [(16, 16), (32, 32), (64, 64), (128, 128)]
    tessellation_results = tensor_model.analyze_dual_tessellation(config, tessellation_factors)
    
    print("Dual Tessellation Analysis:")
    for result in tessellation_results:
        factor = result['tessellation_factor']
        efficiency = result['overall_efficiency']
        print(f"  {factor[0]}x{factor[1]}: {efficiency*100:.2f}% efficiency")
    print()
    
    return results

def demo_memory_analysis():
    """Demonstrate memory performance analysis"""
    print("=== Memory Analysis Demo ===")
    
    hardware = HardwareSpec()
    memory_model = MemoryModel(hardware)
    
    config = StencilConfig(
        shape=StencilShape.BOX_3D1R,
        dimensions=(128, 128, 128),
        iterations=10,
        radius=1,
        halo_size=1
    )
    
    # Analyze memory performance
    results = memory_model.predict_memory_performance(config)
    
    print(f"Configuration: {config.shape.value} {config.dimensions}")
    print(f"Memory bound time: {results['memory_bound_time_seconds']:.6f} seconds")
    print(f"Memory efficiency: {results['memory_efficiency']*100:.2f}%")
    
    cache_behavior = results['cache_behavior']
    print(f"L1 hit rate: {cache_behavior['l1_hit_rate']*100:.2f}%")
    print(f"L2 hit rate: {cache_behavior['l2_hit_rate']*100:.2f}%")
    print(f"Spatial locality: {cache_behavior['spatial_locality']*100:.2f}%")
    print(f"Temporal locality: {cache_behavior['temporal_locality']*100:.2f}%")
    
    footprint = results['memory_footprint']
    print(f"Working set size: {footprint['working_set_bytes']/1e6:.2f} MB")
    
    bandwidth_util = results['bandwidth_utilization']
    print(f"Bandwidth utilization: {bandwidth_util['bandwidth_utilization']*100:.2f}%")
    print()
    
    return results

def demo_roofline_analysis():
    """Demonstrate roofline model analysis"""
    print("=== Roofline Analysis Demo ===")
    
    hardware = HardwareSpec()
    tensor_config = TensorCoreConfig()
    roofline_model = RooflineModel(hardware, tensor_config)
    
    # Create multiple configurations for comparison
    configs = [
        StencilConfig(StencilShape.STAR_1D1R, (8192,), 100, 1, 1),
        StencilConfig(StencilShape.BOX_2D1R, (512, 512), 100, 1, 1),
        StencilConfig(StencilShape.STAR_2D1R, (1024, 1024), 100, 1, 1),
        StencilConfig(StencilShape.BOX_3D1R, (64, 64, 64), 100, 1, 1),
    ]
    
    # Analyze each configuration
    points = []
    for config in configs:
        point = roofline_model.analyze_stencil_config(config)
        points.append(point)
        
        print(f"{point.label}:")
        print(f"  Arithmetic Intensity: {point.arithmetic_intensity:.3f} FLOPS/Byte")
        print(f"  Achieved Performance: {point.performance:.2f} GFLOPS")
        
        expected_perf = roofline_model.calculate_roofline_performance(point.arithmetic_intensity)
        efficiency = point.performance / expected_perf * 100
        print(f"  Roofline Efficiency: {efficiency:.1f}%")
        
        bottleneck = "Memory" if point.arithmetic_intensity < roofline_model.ridge_point else "Compute"
        print(f"  Bottleneck: {bottleneck}")
        print()
    
    # Generate roofline plot
    try:
        fig = roofline_model.generate_roofline_plot(points, "results/roofline_demo.png")
        print("Roofline plot saved to results/roofline_demo.png")
    except Exception as e:
        print(f"Could not generate roofline plot: {e}")
    
    # Generate analysis report
    report = roofline_model.generate_analysis_report(points)
    with open("results/roofline_report.md", "w") as f:
        f.write(report)
    print("Roofline analysis report saved to results/roofline_report.md")
    print()
    
    return points

def demo_scaling_analysis():
    """Demonstrate scaling behavior analysis"""
    print("=== Scaling Analysis Demo ===")
    
    hardware = HardwareSpec()
    tensor_config = TensorCoreConfig()
    perf_model = PerformanceModel(hardware, tensor_config)
    
    # Base configuration
    base_config = StencilConfig(
        shape=StencilShape.BOX_2D1R,
        dimensions=(256, 256),
        iterations=100,
        radius=1,
        halo_size=1
    )
    
    # Analyze scaling behavior
    scale_factors = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    scaling_results = perf_model.analyze_scaling(base_config, scale_factors)
    
    print(f"Base configuration: {base_config.shape.value} {base_config.dimensions}")
    print("Scaling Analysis Results:")
    print("Scale Factor | Problem Size | Throughput (GFLOPS) | Efficiency (%)")
    print("-" * 70)
    
    for result in scaling_results:
        scale = result['scale_factor']
        throughput = result['achieved_flops_per_second'] / 1e9
        efficiency = result['efficiency_percent']
        problem_size = result.get('total_flops', 0) // result.get('iterations', 1)
        
        print(f"{scale:11.1f} | {problem_size:11,} | {throughput:17.2f} | {efficiency:12.1f}")
    
    print()
    return scaling_results

def demo_visualization():
    """Demonstrate visualization capabilities"""
    print("=== Visualization Demo ===")
    
    # Create some sample data for visualization
    visualizer = PerformanceVisualizer("results")
    
    # Sample comparison data
    comparison_data = {
        "ConvStencil": [
            {"throughput": 1250.0, "efficiency": 78.5, "execution_time": 0.0032},
            {"throughput": 1180.0, "efficiency": 74.2, "execution_time": 0.0034},
        ],
        "cuDNN": [
            {"throughput": 980.0, "efficiency": 61.5, "execution_time": 0.0041},
            {"throughput": 950.0, "efficiency": 59.8, "execution_time": 0.0042},
        ]
    }
    
    try:
        visualizer.plot_performance_comparison(comparison_data, "demo_comparison.png")
        print("Performance comparison plot saved to results/demo_comparison.png")
    except Exception as e:
        print(f"Could not generate comparison plot: {e}")
    
    # Sample scaling data
    scaling_data = [
        {"scale_factor": 0.5, "achieved_flops_per_second": 500e9, "efficiency_percent": 65.0, "memory_traffic_bytes": 1e9},
        {"scale_factor": 1.0, "achieved_flops_per_second": 1000e9, "efficiency_percent": 75.0, "memory_traffic_bytes": 2e9},
        {"scale_factor": 2.0, "achieved_flops_per_second": 1800e9, "efficiency_percent": 70.0, "memory_traffic_bytes": 4e9},
        {"scale_factor": 4.0, "achieved_flops_per_second": 3200e9, "efficiency_percent": 60.0, "memory_traffic_bytes": 8e9},
    ]
    
    try:
        visualizer.plot_scaling_analysis(scaling_data, "demo_scaling.png")
        print("Scaling analysis plot saved to results/demo_scaling.png")
    except Exception as e:
        print(f"Could not generate scaling plot: {e}")
    
    print()

def main():
    """Main demo function"""
    print("ConvStencil Modeling Analysis Demo")
    print("==================================")
    print()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run all demos
    try:
        demo_basic_performance_analysis()
        demo_tensor_core_analysis()
        demo_memory_analysis()
        demo_roofline_analysis()
        demo_scaling_analysis()
        demo_visualization()
        
        print("Demo completed successfully!")
        print("Check the 'results/' directory for generated plots and reports.")
        
    except Exception as e:
        print(f"Demo encountered an error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()