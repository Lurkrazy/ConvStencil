"""
Roofline Performance Model

This module implements the Roofline model for analyzing the performance
characteristics of ConvStencil operations on Tensor Cores.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from performance_model import StencilConfig, HardwareSpec
from tensor_core_model import TensorCoreModel, TensorCoreConfig
from memory_model import MemoryModel

@dataclass
class RooflinePoint:
    """Represents a point on the roofline plot"""
    arithmetic_intensity: float
    performance: float  # GFLOPS
    label: str
    config: Optional[StencilConfig] = None

class RooflineModel:
    """Roofline performance analysis for ConvStencil"""
    
    def __init__(self, hardware: HardwareSpec, tensor_config: TensorCoreConfig):
        self.hardware = hardware
        self.tensor_config = tensor_config
        
        # Calculate theoretical peaks
        self.peak_compute_performance = self._calculate_peak_compute()  # GFLOPS
        self.peak_memory_bandwidth = hardware.memory_bandwidth / 1e9  # GB/s
        
        # Ridge point (where memory and compute roofs intersect)
        self.ridge_point = self.peak_compute_performance / self.peak_memory_bandwidth
    
    def _calculate_peak_compute(self) -> float:
        """Calculate theoretical peak compute performance"""
        # Tensor cores: 108 SMs * frequency * ops per tensor core per cycle
        tensor_ops_per_cycle = self.tensor_config.m * self.tensor_config.n * self.tensor_config.k
        ops_per_warp = tensor_ops_per_cycle
        warps_per_sm = 4  # Typical concurrent warps doing tensor operations
        
        peak_ops_per_second = (self.hardware.tensor_core_count * 
                              self.hardware.tensor_core_frequency * 
                              ops_per_warp * warps_per_sm / self.tensor_config.warp_size)
        
        return peak_ops_per_second / 1e9  # Convert to GFLOPS
    
    def calculate_roofline_performance(self, arithmetic_intensity: float) -> float:
        """Calculate expected performance from roofline model"""
        
        # Memory-bound performance
        memory_bound_perf = arithmetic_intensity * self.peak_memory_bandwidth
        
        # Compute-bound performance (with efficiency factor)
        compute_efficiency = 0.85  # Realistic efficiency for tensor cores
        compute_bound_perf = self.peak_compute_performance * compute_efficiency
        
        # Actual performance is minimum of the two
        return min(memory_bound_perf, compute_bound_perf)
    
    def analyze_stencil_config(self, config: StencilConfig) -> RooflinePoint:
        """Analyze a single stencil configuration"""
        
        # Calculate arithmetic intensity
        from performance_model import PerformanceModel
        perf_model = PerformanceModel(self.hardware, self.tensor_config)
        
        # Get detailed analysis
        analysis = perf_model.predict_performance(config)
        arithmetic_intensity = analysis["arithmetic_intensity"]
        achieved_performance = analysis["achieved_flops_per_second"] / 1e9  # GFLOPS
        
        return RooflinePoint(
            arithmetic_intensity=arithmetic_intensity,
            performance=achieved_performance,
            label=f"{config.shape.value}_{config.dimensions}",
            config=config
        )
    
    def analyze_multiple_configs(self, configs: List[StencilConfig]) -> List[RooflinePoint]:
        """Analyze multiple configurations"""
        points = []
        for config in configs:
            point = self.analyze_stencil_config(config)
            points.append(point)
        return points
    
    def generate_roofline_plot(self, points: List[RooflinePoint], 
                              save_path: Optional[str] = None) -> plt.Figure:
        """Generate roofline plot with analysis points"""
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Generate roofline curves
        ai_range = np.logspace(-2, 2, 1000)  # Arithmetic intensity range
        
        # Memory-bound roof
        memory_roof = ai_range * self.peak_memory_bandwidth
        
        # Compute-bound roof
        compute_roof = np.full_like(ai_range, self.peak_compute_performance * 0.85)
        
        # Combined roofline
        roofline = np.minimum(memory_roof, compute_roof)
        
        # Plot rooflines
        ax.loglog(ai_range, memory_roof, 'r--', linewidth=2, alpha=0.7, 
                 label=f'Memory Bound ({self.peak_memory_bandwidth:.0f} GB/s)')
        ax.loglog(ai_range, compute_roof, 'b--', linewidth=2, alpha=0.7,
                 label=f'Compute Bound ({self.peak_compute_performance*0.85:.0f} GFLOPS)')
        ax.loglog(ai_range, roofline, 'k-', linewidth=3, label='Roofline')
        
        # Plot ridge point
        ridge_perf = self.calculate_roofline_performance(self.ridge_point)
        ax.loglog(self.ridge_point, ridge_perf, 'ko', markersize=10, 
                 label=f'Ridge Point (AI={self.ridge_point:.2f})')
        
        # Plot analysis points
        colors = plt.cm.Set1(np.linspace(0, 1, len(points)))
        for point, color in zip(points, colors):
            ax.loglog(point.arithmetic_intensity, point.performance, 
                     'o', markersize=8, color=color, label=point.label)
            
            # Add efficiency annotation
            expected_perf = self.calculate_roofline_performance(point.arithmetic_intensity)
            efficiency = point.performance / expected_perf * 100
            ax.annotate(f'{efficiency:.1f}%', 
                       (point.arithmetic_intensity, point.performance),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # Formatting
        ax.set_xlabel('Arithmetic Intensity (FLOPS/Byte)', fontsize=12)
        ax.set_ylabel('Performance (GFLOPS)', fontsize=12)
        ax.set_title('ConvStencil Roofline Analysis', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Set reasonable limits
        ax.set_xlim(0.01, 100)
        ax.set_ylim(1, self.peak_compute_performance * 2)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Roofline plot saved to {save_path}")
        
        return fig
    
    def analyze_bottlenecks(self, points: List[RooflinePoint]) -> Dict[str, List[RooflinePoint]]:
        """Categorize configurations by bottleneck type"""
        
        memory_bound = []
        compute_bound = []
        
        for point in points:
            if point.arithmetic_intensity < self.ridge_point:
                memory_bound.append(point)
            else:
                compute_bound.append(point)
        
        return {
            "memory_bound": memory_bound,
            "compute_bound": compute_bound
        }
    
    def suggest_optimizations(self, points: List[RooflinePoint]) -> Dict[str, List[str]]:
        """Suggest optimizations based on roofline analysis"""
        
        bottlenecks = self.analyze_bottlenecks(points)
        suggestions = {}
        
        for point in bottlenecks["memory_bound"]:
            point_suggestions = [
                "Increase arithmetic intensity by fusing operations",
                "Optimize data layout for better cache utilization",
                "Use blocking/tiling to improve temporal locality",
                "Consider data compression techniques"
            ]
            suggestions[point.label] = point_suggestions
        
        for point in bottlenecks["compute_bound"]:
            point_suggestions = [
                "Optimize tensor core utilization",
                "Improve stencil2row transformation efficiency",
                "Use dual tessellation for better parallelism",
                "Optimize kernel fusion strategies"
            ]
            suggestions[point.label] = point_suggestions
        
        return suggestions
    
    def efficiency_analysis(self, points: List[RooflinePoint]) -> Dict[str, float]:
        """Analyze efficiency of different configurations"""
        
        efficiencies = {}
        
        for point in points:
            expected_perf = self.calculate_roofline_performance(point.arithmetic_intensity)
            efficiency = point.performance / expected_perf
            efficiencies[point.label] = efficiency
        
        return efficiencies
    
    def generate_analysis_report(self, points: List[RooflinePoint]) -> str:
        """Generate comprehensive roofline analysis report"""
        
        report = "# ConvStencil Roofline Analysis Report\n\n"
        
        # Hardware specifications
        report += "## Hardware Specifications\n\n"
        report += f"- Peak Compute Performance: {self.peak_compute_performance:.0f} GFLOPS\n"
        report += f"- Peak Memory Bandwidth: {self.peak_memory_bandwidth:.0f} GB/s\n"
        report += f"- Ridge Point: {self.ridge_point:.2f} FLOPS/Byte\n\n"
        
        # Configuration analysis
        report += "## Configuration Analysis\n\n"
        for point in points:
            expected_perf = self.calculate_roofline_performance(point.arithmetic_intensity)
            efficiency = point.performance / expected_perf * 100
            bottleneck = "Memory" if point.arithmetic_intensity < self.ridge_point else "Compute"
            
            report += f"### {point.label}\n"
            report += f"- Arithmetic Intensity: {point.arithmetic_intensity:.3f} FLOPS/Byte\n"
            report += f"- Achieved Performance: {point.performance:.2f} GFLOPS\n"
            report += f"- Expected Performance: {expected_perf:.2f} GFLOPS\n"
            report += f"- Efficiency: {efficiency:.1f}%\n"
            report += f"- Bottleneck: {bottleneck}\n\n"
        
        # Bottleneck summary
        bottlenecks = self.analyze_bottlenecks(points)
        report += "## Bottleneck Summary\n\n"
        report += f"- Memory-bound configurations: {len(bottlenecks['memory_bound'])}\n"
        report += f"- Compute-bound configurations: {len(bottlenecks['compute_bound'])}\n\n"
        
        # Optimization suggestions
        suggestions = self.suggest_optimizations(points)
        report += "## Optimization Suggestions\n\n"
        
        for config, opts in suggestions.items():
            report += f"### {config}\n"
            for opt in opts:
                report += f"- {opt}\n"
            report += "\n"
        
        return report