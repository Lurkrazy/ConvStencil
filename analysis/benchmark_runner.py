"""
Benchmarking Framework

This module provides automated benchmarking tools for ConvStencil
performance evaluation and comparison with other methods.
"""

import subprocess
import time
import json
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import yaml

from performance_model import StencilConfig, StencilShape, HardwareSpec
from tensor_core_model import TensorCoreModel, TensorCoreConfig
from memory_model import MemoryModel

@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs"""
    name: str
    executable: str
    shapes: List[str]
    sizes: List[Tuple[int, ...]]
    iterations: List[int]
    warmup_runs: int = 3
    measurement_runs: int = 10
    output_dir: str = "results"

@dataclass
class BenchmarkResult:
    """Results from a benchmark run"""
    config: Dict
    execution_time: float
    std_deviation: float
    throughput: float  # GFLOPS
    bandwidth: float   # GB/s
    efficiency: float  # % of peak
    error_code: int = 0
    error_message: str = ""

class BenchmarkRunner:
    """Automated benchmarking framework for ConvStencil"""
    
    def __init__(self, build_dir: str = "../build"):
        self.build_dir = Path(build_dir)
        self.results_dir = Path("results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize models for analysis
        self.hardware = HardwareSpec()
        self.tensor_config = TensorCoreConfig()
        self.memory_model = MemoryModel(self.hardware)
        self.tensor_model = TensorCoreModel(self.hardware, self.tensor_config)
    
    def run_single_benchmark(self, executable: str, shape: str, 
                           dimensions: Tuple[int, ...], iterations: int) -> BenchmarkResult:
        """Run a single benchmark configuration"""
        
        # Construct command
        cmd = [str(self.build_dir / executable), shape] + [str(d) for d in dimensions] + [str(iterations)]
        
        print(f"Running: {' '.join(cmd)}")
        
        # Warmup runs
        for _ in range(3):
            try:
                subprocess.run(cmd, capture_output=True, timeout=60)
            except subprocess.TimeoutExpired:
                print(f"Warmup timeout for {shape} {dimensions}")
        
        # Measurement runs
        times = []
        for run in range(10):
            try:
                start_time = time.time()
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                end_time = time.time()
                
                if result.returncode == 0:
                    # Parse execution time from output if available
                    execution_time = self._parse_execution_time(result.stdout)
                    if execution_time is None:
                        execution_time = end_time - start_time
                    times.append(execution_time)
                else:
                    print(f"Error in run {run}: {result.stderr}")
                    return BenchmarkResult(
                        config={"shape": shape, "dimensions": dimensions, "iterations": iterations},
                        execution_time=0,
                        std_deviation=0,
                        throughput=0,
                        bandwidth=0,
                        efficiency=0,
                        error_code=result.returncode,
                        error_message=result.stderr
                    )
            except subprocess.TimeoutExpired:
                print(f"Timeout in run {run}")
                continue
        
        if not times:
            return BenchmarkResult(
                config={"shape": shape, "dimensions": dimensions, "iterations": iterations},
                execution_time=0,
                std_deviation=0,
                throughput=0,
                bandwidth=0,
                efficiency=0,
                error_code=-1,
                error_message="All runs failed"
            )
        
        # Calculate statistics
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        # Calculate performance metrics
        metrics = self._calculate_performance_metrics(shape, dimensions, iterations, mean_time)
        
        return BenchmarkResult(
            config={"shape": shape, "dimensions": dimensions, "iterations": iterations},
            execution_time=mean_time,
            std_deviation=std_time,
            throughput=metrics["throughput"],
            bandwidth=metrics["bandwidth"],
            efficiency=metrics["efficiency"]
        )
    
    def _parse_execution_time(self, output: str) -> Optional[float]:
        """Parse execution time from program output"""
        lines = output.strip().split('\n')
        for line in lines:
            if "time" in line.lower() and ("ms" in line or "seconds" in line):
                # Try to extract numerical value
                import re
                numbers = re.findall(r'[\d.]+', line)
                if numbers:
                    time_val = float(numbers[0])
                    if "ms" in line:
                        return time_val / 1000.0  # Convert to seconds
                    else:
                        return time_val
        return None
    
    def _calculate_performance_metrics(self, shape: str, dimensions: Tuple[int, ...], 
                                     iterations: int, execution_time: float) -> Dict[str, float]:
        """Calculate performance metrics from execution time"""
        
        # Create stencil config for analysis
        shape_enum = StencilShape(shape.lower())
        config = StencilConfig(
            shape=shape_enum,
            dimensions=dimensions,
            iterations=iterations,
            radius=1 if "1r" in shape else (2 if "2r" in shape else 3),
            halo_size=1 if "1r" in shape else (2 if "2r" in shape else 3)
        )
        
        # Get stencil properties
        if "star_1d" in shape:
            points = 3 if "1r" in shape else 5
        elif "box_2d" in shape or "star_2d" in shape:
            if "1r" in shape:
                points = 9 if "box" in shape else 5
            else:
                points = 49 if "box" in shape else 13
        else:  # 3D
            points = 27 if "box" in shape else 7
        
        # Calculate FLOPS
        total_elements = np.prod(dimensions)
        total_flops = total_elements * points * 2 * iterations  # multiply-add operations
        throughput = total_flops / execution_time / 1e9  # GFLOPS
        
        # Estimate bandwidth
        element_size = 8  # double precision
        data_movement = total_elements * element_size * 2 * iterations  # input + output
        bandwidth = data_movement / execution_time / 1e9  # GB/s
        
        # Calculate efficiency relative to theoretical peak
        peak_flops = self.hardware.tensor_core_count * self.hardware.tensor_core_frequency * 8 * 8 * 4 / 32 / 1e9
        efficiency = (throughput / peak_flops) * 100
        
        return {
            "throughput": throughput,
            "bandwidth": bandwidth,
            "efficiency": efficiency
        }
    
    def run_benchmark_suite(self, config_file: str) -> List[BenchmarkResult]:
        """Run a complete benchmark suite from configuration file"""
        
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        benchmark_config = BenchmarkConfig(**config)
        results = []
        
        total_configs = len(benchmark_config.shapes) * len(benchmark_config.sizes) * len(benchmark_config.iterations)
        current_config = 0
        
        for shape in benchmark_config.shapes:
            # Determine appropriate executable
            if "1d" in shape:
                executable = "convstencil_1d"
            elif "2d" in shape:
                executable = "convstencil_2d"
            elif "3d" in shape:
                executable = "convstencil_3d"
            else:
                continue
            
            for size in benchmark_config.sizes:
                for iterations in benchmark_config.iterations:
                    current_config += 1
                    print(f"Running configuration {current_config}/{total_configs}")
                    
                    result = self.run_single_benchmark(executable, shape, size, iterations)
                    results.append(result)
                    
                    # Save intermediate results
                    self._save_results(results, benchmark_config.output_dir)
        
        return results
    
    def compare_with_cudnn(self, shapes: List[str], sizes: List[Tuple[int, ...]]) -> Dict[str, List[BenchmarkResult]]:
        """Compare ConvStencil performance with cuDNN implementations"""
        
        results = {
            "convstencil": [],
            "cudnn": []
        }
        
        for shape in shapes:
            for size in sizes:
                # Run ConvStencil
                if "1d" in shape:
                    convstencil_exec = "convstencil_1d"
                elif "2d" in shape:
                    convstencil_exec = "convstencil_2d"
                elif "3d" in shape:
                    convstencil_exec = "convstencil_3d"
                else:
                    continue
                
                convstencil_result = self.run_single_benchmark(convstencil_exec, shape, size, 100)
                results["convstencil"].append(convstencil_result)
                
                # Run cuDNN equivalent
                cudnn_exec = self._get_cudnn_executable(shape)
                if cudnn_exec:
                    cudnn_result = self.run_single_benchmark(cudnn_exec, shape, size, 100)
                    results["cudnn"].append(cudnn_result)
        
        return results
    
    def _get_cudnn_executable(self, shape: str) -> Optional[str]:
        """Get corresponding cuDNN executable for a given shape"""
        cudnn_mapping = {
            "star_1d1r": "cudnn_1d3p",
            "star_1d2r": "cudnn_1d5p",
            "box_2d1r": "cudnn_box2d9p",
            "box_2d3r": "cudnn_box2d49p",
            "box_3d1r": "cudnn_box3d27p"
        }
        return cudnn_mapping.get(shape)
    
    def _save_results(self, results: List[BenchmarkResult], output_dir: str):
        """Save benchmark results to file"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Convert results to JSON-serializable format
        json_results = [asdict(result) for result in results]
        
        # Save to file
        with open(output_path / "benchmark_results.json", 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"Results saved to {output_path / 'benchmark_results.json'}")
    
    def generate_performance_report(self, results: List[BenchmarkResult]) -> str:
        """Generate a performance analysis report"""
        
        report = "# ConvStencil Performance Analysis Report\n\n"
        
        # Summary statistics
        successful_runs = [r for r in results if r.error_code == 0]
        failed_runs = [r for r in results if r.error_code != 0]
        
        report += f"## Summary\n\n"
        report += f"- Total configurations: {len(results)}\n"
        report += f"- Successful runs: {len(successful_runs)}\n"
        report += f"- Failed runs: {len(failed_runs)}\n\n"
        
        if successful_runs:
            throughputs = [r.throughput for r in successful_runs]
            efficiencies = [r.efficiency for r in successful_runs]
            
            report += f"## Performance Metrics\n\n"
            report += f"- Average throughput: {np.mean(throughputs):.2f} GFLOPS\n"
            report += f"- Peak throughput: {np.max(throughputs):.2f} GFLOPS\n"
            report += f"- Average efficiency: {np.mean(efficiencies):.2f}%\n"
            report += f"- Peak efficiency: {np.max(efficiencies):.2f}%\n\n"
            
            # Per-shape analysis
            shapes = list(set(r.config["shape"] for r in successful_runs))
            report += f"## Per-Shape Analysis\n\n"
            
            for shape in shapes:
                shape_results = [r for r in successful_runs if r.config["shape"] == shape]
                shape_throughputs = [r.throughput for r in shape_results]
                shape_efficiencies = [r.efficiency for r in shape_results]
                
                report += f"### {shape}\n"
                report += f"- Runs: {len(shape_results)}\n"
                report += f"- Average throughput: {np.mean(shape_throughputs):.2f} GFLOPS\n"
                report += f"- Average efficiency: {np.mean(shape_efficiencies):.2f}%\n\n"
        
        if failed_runs:
            report += f"## Failed Configurations\n\n"
            for run in failed_runs:
                report += f"- {run.config}: Error {run.error_code} - {run.error_message}\n"
        
        return report