"""
Main Performance Analysis Script

This script provides a comprehensive performance analysis of ConvStencil
including modeling, benchmarking, and comparison with theoretical predictions.
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List

from performance_model import PerformanceModel, StencilConfig, StencilShape, HardwareSpec, TensorCoreConfig
from tensor_core_model import TensorCoreModel
from memory_model import MemoryModel
from benchmark_runner import BenchmarkRunner

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="ConvStencil Performance Analysis")
    
    parser.add_argument("--shape", type=str, required=True,
                       choices=["star_1d1r", "star_1d2r", "box_2d1r", "star_2d1r", 
                               "box_2d3r", "star_2d3r", "box_3d1r", "star_3d1r"],
                       help="Stencil shape to analyze")
    
    parser.add_argument("--size", type=int, nargs="+", required=True,
                       help="Problem size dimensions")
    
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of time iterations")
    
    parser.add_argument("--mode", type=str, default="analyze",
                       choices=["analyze", "benchmark", "compare", "model_validation"],
                       help="Analysis mode")
    
    parser.add_argument("--build-dir", type=str, default="../build",
                       help="Build directory path")
    
    parser.add_argument("--output-dir", type=str, default="results",
                       help="Output directory for results")
    
    parser.add_argument("--scaling-analysis", action="store_true",
                       help="Perform scaling analysis")
    
    parser.add_argument("--save-plots", action="store_true",
                       help="Save performance plots")
    
    return parser.parse_args()

def create_stencil_config(shape: str, size: List[int], iterations: int) -> StencilConfig:
    """Create StencilConfig from command line arguments"""
    shape_enum = StencilShape(shape.lower())
    
    # Determine radius based on shape
    if "1r" in shape:
        radius = 1
    elif "2r" in shape:
        radius = 2
    elif "3r" in shape:
        radius = 3
    else:
        radius = 1
    
    return StencilConfig(
        shape=shape_enum,
        dimensions=tuple(size),
        iterations=iterations,
        radius=radius,
        halo_size=radius
    )

def analyze_performance(config: StencilConfig, hardware: HardwareSpec, 
                       tensor_config: TensorCoreConfig) -> Dict:
    """Perform comprehensive performance analysis"""
    
    # Initialize models
    perf_model = PerformanceModel(hardware, tensor_config)
    tensor_model = TensorCoreModel(hardware, tensor_config)
    memory_model = MemoryModel(hardware)
    
    # Run analysis
    print("Running performance modeling...")
    
    # Core performance prediction
    performance = perf_model.predict_performance(config)
    
    # Tensor core specific analysis
    tensor_analysis = tensor_model.predict_tensor_core_performance(config)
    
    # Memory analysis
    memory_analysis = memory_model.predict_memory_performance(config)
    
    # Combine results
    analysis_results = {
        "configuration": {
            "shape": config.shape.value,
            "dimensions": config.dimensions,
            "iterations": config.iterations,
            "radius": config.radius
        },
        "performance_prediction": performance,
        "tensor_core_analysis": tensor_analysis,
        "memory_analysis": memory_analysis
    }
    
    return analysis_results

def run_scaling_analysis(base_config: StencilConfig, hardware: HardwareSpec,
                        tensor_config: TensorCoreConfig) -> Dict:
    """Run scaling analysis with different problem sizes"""
    
    print("Running scaling analysis...")
    
    perf_model = PerformanceModel(hardware, tensor_config)
    scale_factors = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    scaling_results = perf_model.analyze_scaling(base_config, scale_factors)
    
    return {
        "base_config": {
            "shape": base_config.shape.value,
            "dimensions": base_config.dimensions,
            "iterations": base_config.iterations
        },
        "scaling_data": scaling_results
    }

def run_benchmarks(config: StencilConfig, build_dir: str) -> Dict:
    """Run actual benchmarks"""
    
    print("Running benchmarks...")
    
    runner = BenchmarkRunner(build_dir)
    
    # Run single configuration
    if "1d" in config.shape.value:
        executable = "convstencil_1d"
    elif "2d" in config.shape.value:
        executable = "convstencil_2d"
    elif "3d" in config.shape.value:
        executable = "convstencil_3d"
    else:
        raise ValueError(f"Unknown shape type: {config.shape.value}")
    
    result = runner.run_single_benchmark(
        executable, config.shape.value, config.dimensions, config.iterations
    )
    
    return {
        "benchmark_result": result.__dict__,
        "configuration": {
            "shape": config.shape.value,
            "dimensions": config.dimensions,
            "iterations": config.iterations
        }
    }

def compare_prediction_with_measurement(analysis_results: Dict, benchmark_results: Dict) -> Dict:
    """Compare model predictions with actual measurements"""
    
    predicted = analysis_results["performance_prediction"]
    measured = benchmark_results["benchmark_result"]
    
    if measured["error_code"] != 0:
        return {
            "comparison_status": "failed",
            "error": measured["error_message"]
        }
    
    # Calculate accuracy metrics
    predicted_time = predicted["predicted_time_seconds"]
    measured_time = measured["execution_time"]
    
    time_error = abs(predicted_time - measured_time) / measured_time * 100
    
    predicted_throughput = predicted["achieved_flops_per_second"] / 1e9  # GFLOPS
    measured_throughput = measured["throughput"]
    
    throughput_error = abs(predicted_throughput - measured_throughput) / measured_throughput * 100
    
    return {
        "comparison_status": "success",
        "predicted_time_seconds": predicted_time,
        "measured_time_seconds": measured_time,
        "time_prediction_error_percent": time_error,
        "predicted_throughput_gflops": predicted_throughput,
        "measured_throughput_gflops": measured_throughput,
        "throughput_prediction_error_percent": throughput_error,
        "model_accuracy": "good" if time_error < 20 else "poor"
    }

def generate_plots(results: Dict, output_dir: str):
    """Generate performance analysis plots"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Plot 1: Performance breakdown
    if "performance_prediction" in results:
        perf = results["performance_prediction"]
        
        plt.figure(figsize=(10, 6))
        
        # Efficiency breakdown
        plt.subplot(2, 2, 1)
        efficiency_data = [
            perf["efficiency_percent"],
            perf["tensor_core_utilization"] * 100,
            perf["arithmetic_intensity"] * 10  # Scale for visibility
        ]
        labels = ["Overall Efficiency", "Tensor Core Util.", "Arith. Intensity (x10)"]
        plt.bar(labels, efficiency_data)
        plt.title("Performance Efficiency Breakdown")
        plt.ylabel("Percentage")
        plt.xticks(rotation=45)
        
        # Memory vs Compute
        plt.subplot(2, 2, 2)
        times = [perf["compute_bound_time"], perf["memory_bound_time"]]
        labels = ["Compute Time", "Memory Time"]
        colors = ["blue", "red"]
        plt.bar(labels, times, color=colors)
        plt.title("Compute vs Memory Bound Analysis")
        plt.ylabel("Time (seconds)")
        
        # Bottleneck indication
        plt.subplot(2, 2, 3)
        bottleneck_text = f"Bottleneck: {perf['bottleneck'].upper()}"
        plt.text(0.5, 0.5, bottleneck_text, ha='center', va='center', 
                fontsize=16, transform=plt.gca().transAxes)
        plt.axis('off')
        plt.title("Performance Bottleneck")
        
        plt.tight_layout()
        plt.savefig(output_path / "performance_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    # Plot 2: Scaling analysis
    if "scaling_data" in results:
        scaling_data = results["scaling_data"]
        
        plt.figure(figsize=(12, 8))
        
        scale_factors = [d["scale_factor"] for d in scaling_data]
        throughputs = [d["achieved_flops_per_second"] / 1e9 for d in scaling_data]  # GFLOPS
        efficiencies = [d["efficiency_percent"] for d in scaling_data]
        
        plt.subplot(2, 2, 1)
        plt.plot(scale_factors, throughputs, 'o-', linewidth=2, markersize=8)
        plt.xlabel("Scale Factor")
        plt.ylabel("Throughput (GFLOPS)")
        plt.title("Throughput Scaling")
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 2, 2)
        plt.plot(scale_factors, efficiencies, 'o-', linewidth=2, markersize=8, color='red')
        plt.xlabel("Scale Factor")
        plt.ylabel("Efficiency (%)")
        plt.title("Efficiency Scaling")
        plt.grid(True, alpha=0.3)
        
        # Performance vs problem size
        problem_sizes = [np.prod(d["dimensions"]) if "dimensions" in d else 0 for d in scaling_data]
        if any(problem_sizes):
            plt.subplot(2, 2, 3)
            plt.loglog(problem_sizes, throughputs, 'o-', linewidth=2, markersize=8)
            plt.xlabel("Problem Size (elements)")
            plt.ylabel("Throughput (GFLOPS)")
            plt.title("Throughput vs Problem Size")
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path / "scaling_analysis.png", dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"Plots saved to {output_path}")

def save_results(results: Dict, output_dir: str):
    """Save analysis results to JSON file"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "analysis_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_path / 'analysis_results.json'}")

def print_summary(results: Dict):
    """Print analysis summary"""
    
    print("\n" + "="*60)
    print("CONVSTENCIL PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)
    
    if "configuration" in results:
        config = results["configuration"]
        print(f"Configuration: {config['shape']} {config['dimensions']} ({config['iterations']} iterations)")
    
    if "performance_prediction" in results:
        perf = results["performance_prediction"]
        print(f"\nPerformance Prediction:")
        print(f"  Predicted time: {perf['predicted_time_seconds']:.6f} seconds")
        print(f"  Throughput: {perf['achieved_flops_per_second']/1e9:.2f} GFLOPS")
        print(f"  Efficiency: {perf['efficiency_percent']:.2f}%")
        print(f"  Bottleneck: {perf['bottleneck']}")
        print(f"  Tensor Core Utilization: {perf['tensor_core_utilization']*100:.2f}%")
        print(f"  Arithmetic Intensity: {perf['arithmetic_intensity']:.2f}")
    
    if "tensor_core_analysis" in results:
        tensor = results["tensor_core_analysis"]
        print(f"\nTensor Core Analysis:")
        print(f"  Tensor Core Efficiency: {tensor['tensor_core_efficiency']*100:.2f}%")
        print(f"  Achieved Performance: {tensor['achieved_tflops']:.2f} TFLOPS")
    
    if "memory_analysis" in results:
        memory = results["memory_analysis"]
        print(f"\nMemory Analysis:")
        print(f"  Memory Efficiency: {memory['memory_efficiency']*100:.2f}%")
        print(f"  L1 Hit Rate: {memory['cache_behavior']['l1_hit_rate']*100:.2f}%")
        print(f"  L2 Hit Rate: {memory['cache_behavior']['l2_hit_rate']*100:.2f}%")
    
    if "comparison_status" in results and results["comparison_status"] == "success":
        print(f"\nModel Validation:")
        print(f"  Time Prediction Error: {results['time_prediction_error_percent']:.2f}%")
        print(f"  Throughput Prediction Error: {results['throughput_prediction_error_percent']:.2f}%")
        print(f"  Model Accuracy: {results['model_accuracy']}")
    
    print("="*60)

def main():
    """Main analysis function"""
    
    args = parse_arguments()
    
    # Create configuration
    config = create_stencil_config(args.shape, args.size, args.iterations)
    hardware = HardwareSpec()
    tensor_config = TensorCoreConfig()
    
    print(f"Analyzing ConvStencil performance for {args.shape} {args.size}")
    
    results = {}
    
    # Run analysis based on mode
    if args.mode in ["analyze", "model_validation"]:
        analysis_results = analyze_performance(config, hardware, tensor_config)
        results.update(analysis_results)
        
        if args.scaling_analysis:
            scaling_results = run_scaling_analysis(config, hardware, tensor_config)
            results.update(scaling_results)
    
    if args.mode in ["benchmark", "model_validation", "compare"]:
        benchmark_results = run_benchmarks(config, args.build_dir)
        results.update(benchmark_results)
        
        if args.mode == "model_validation" and "performance_prediction" in results:
            comparison = compare_prediction_with_measurement(results, benchmark_results)
            results.update(comparison)
    
    # Generate outputs
    print_summary(results)
    save_results(results, args.output_dir)
    
    if args.save_plots:
        generate_plots(results, args.output_dir)
    
    print(f"\nAnalysis complete. Results saved to {args.output_dir}/")

if __name__ == "__main__":
    main()