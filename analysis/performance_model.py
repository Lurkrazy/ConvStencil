"""
ConvStencil Performance Model

This module implements the core performance modeling framework for ConvStencil,
providing mathematical models to predict performance on Tensor Cores.
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class StencilShape(Enum):
    """Supported stencil shapes"""
    STAR_1D1R = "star_1d1r"
    STAR_1D2R = "star_1d2r"
    BOX_2D1R = "box_2d1r"
    STAR_2D1R = "star_2d1r"
    BOX_2D3R = "box_2d3r"
    STAR_2D3R = "star_2d3r"
    BOX_3D1R = "box_3d1r"
    STAR_3D1R = "star_3d1r"

@dataclass
class HardwareSpec:
    """Hardware specifications for modeling"""
    tensor_core_count: int = 108  # A100 typical
    tensor_core_frequency: float = 1.4e9  # Hz
    memory_bandwidth: float = 1555e9  # bytes/s for A100
    l2_cache_size: int = 40 * 1024 * 1024  # 40MB
    shared_memory_size: int = 164 * 1024  # 164KB per SM
    compute_capability: str = "8.0"
    
@dataclass
class StencilConfig:
    """Stencil computation configuration"""
    shape: StencilShape
    dimensions: Tuple[int, ...]  # (height, width) or (height, width, depth)
    iterations: int
    radius: int
    halo_size: int
    data_type: str = "double"  # "float" or "double"

@dataclass
class TensorCoreConfig:
    """Tensor Core specific configuration"""
    m: int = 8  # Matrix dimensions for tensor core
    n: int = 8
    k: int = 4  # for double precision
    warp_size: int = 32
    warps_per_block: int = 8

class PerformanceModel:
    """Core performance modeling framework for ConvStencil"""
    
    def __init__(self, hardware: HardwareSpec, tensor_config: TensorCoreConfig):
        self.hardware = hardware
        self.tensor_config = tensor_config
        self.data_size = 8 if hardware.compute_capability >= "8.0" else 4  # bytes per element (double/float)
    
    def get_stencil_properties(self, shape: StencilShape) -> Dict[str, int]:
        """Get properties of different stencil shapes"""
        properties = {
            StencilShape.STAR_1D1R: {"points": 3, "radius": 1, "dims": 1},
            StencilShape.STAR_1D2R: {"points": 5, "radius": 2, "dims": 1},
            StencilShape.BOX_2D1R: {"points": 9, "radius": 1, "dims": 2},
            StencilShape.STAR_2D1R: {"points": 5, "radius": 1, "dims": 2},
            StencilShape.BOX_2D3R: {"points": 49, "radius": 3, "dims": 2},
            StencilShape.STAR_2D3R: {"points": 13, "radius": 3, "dims": 2},
            StencilShape.BOX_3D1R: {"points": 27, "radius": 1, "dims": 3},
            StencilShape.STAR_3D1R: {"points": 7, "radius": 1, "dims": 3},
        }
        return properties.get(shape, {"points": 1, "radius": 1, "dims": 1})
    
    def calculate_arithmetic_intensity(self, config: StencilConfig) -> float:
        """Calculate arithmetic intensity (FLOPS per byte)"""
        props = self.get_stencil_properties(config.shape)
        
        # FLOPS per output point
        flops_per_point = props["points"] * 2  # multiply-add operations
        
        # Data movement per point (considering halo regions)
        if props["dims"] == 1:
            data_per_point = (2 * props["radius"] + 1) * self.data_size
        elif props["dims"] == 2:
            halo_factor = (config.dimensions[0] + 2 * props["radius"]) * \
                         (config.dimensions[1] + 2 * props["radius"])
            data_per_point = halo_factor * self.data_size / (config.dimensions[0] * config.dimensions[1])
        else:  # 3D
            halo_factor = (config.dimensions[0] + 2 * props["radius"]) * \
                         (config.dimensions[1] + 2 * props["radius"]) * \
                         (config.dimensions[2] + 2 * props["radius"])
            data_per_point = halo_factor * self.data_size / \
                           (config.dimensions[0] * config.dimensions[1] * config.dimensions[2])
        
        return flops_per_point / data_per_point
    
    def calculate_tensor_core_utilization(self, config: StencilConfig) -> float:
        """Calculate tensor core utilization efficiency"""
        props = self.get_stencil_properties(config.shape)
        
        # Calculate how well the stencil maps to tensor core dimensions
        if props["dims"] == 1:
            total_elements = config.dimensions[0]
        elif props["dims"] == 2:
            total_elements = config.dimensions[0] * config.dimensions[1]
        else:
            total_elements = config.dimensions[0] * config.dimensions[1] * config.dimensions[2]
        
        # Tensor core operates on 8x8 matrices
        tensor_core_ops = self.tensor_config.m * self.tensor_config.n * self.tensor_config.k
        total_tensor_cores = self.hardware.tensor_core_count
        
        # Calculate utilization based on problem size alignment
        ideal_ops = total_elements * props["points"]
        tensor_core_capacity = total_tensor_cores * tensor_core_ops
        
        utilization = min(1.0, ideal_ops / tensor_core_capacity)
        
        # Apply efficiency factors for stencil2row transformation
        transformation_efficiency = self.calculate_transformation_efficiency(config)
        
        return utilization * transformation_efficiency
    
    def calculate_transformation_efficiency(self, config: StencilConfig) -> float:
        """Calculate efficiency of stencil2row transformation"""
        props = self.get_stencil_properties(config.shape)
        
        # Efficiency depends on how well stencil patterns map to matrix multiply
        base_efficiency = 0.85  # Typical efficiency for well-optimized transformations
        
        # Penalty for irregular stencil patterns
        if config.shape in [StencilShape.STAR_2D1R, StencilShape.STAR_3D1R]:
            base_efficiency *= 0.9  # Star patterns have some inefficiency
        
        # Penalty for larger radius (more complex transformations)
        radius_penalty = 1.0 - (props["radius"] - 1) * 0.05
        base_efficiency *= max(0.7, radius_penalty)
        
        return base_efficiency
    
    def calculate_memory_traffic(self, config: StencilConfig) -> float:
        """Calculate total memory traffic in bytes"""
        props = self.get_stencil_properties(config.shape)
        
        if props["dims"] == 1:
            total_elements = config.dimensions[0] + 2 * props["radius"]
        elif props["dims"] == 2:
            total_elements = (config.dimensions[0] + 2 * props["radius"]) * \
                           (config.dimensions[1] + 2 * props["radius"])
        else:
            total_elements = (config.dimensions[0] + 2 * props["radius"]) * \
                           (config.dimensions[1] + 2 * props["radius"]) * \
                           (config.dimensions[2] + 2 * props["radius"])
        
        # Input data + output data + intermediate transformations
        input_traffic = total_elements * self.data_size
        output_traffic = np.prod(config.dimensions) * self.data_size
        transformation_overhead = total_elements * self.data_size * 0.3  # 30% overhead
        
        return (input_traffic + output_traffic + transformation_overhead) * config.iterations
    
    def predict_performance(self, config: StencilConfig) -> Dict[str, float]:
        """Predict performance metrics for given configuration"""
        props = self.get_stencil_properties(config.shape)
        
        # Calculate total FLOPS
        total_points = np.prod(config.dimensions)
        total_flops = total_points * props["points"] * 2 * config.iterations  # multiply-add
        
        # Calculate theoretical peak performance
        peak_flops = self.hardware.tensor_core_count * self.hardware.tensor_core_frequency * \
                    self.tensor_config.m * self.tensor_config.n * self.tensor_config.k / self.tensor_config.warp_size
        
        # Calculate achievable performance
        utilization = self.calculate_tensor_core_utilization(config)
        achievable_flops = peak_flops * utilization
        
        # Calculate memory constraints
        memory_traffic = self.calculate_memory_traffic(config)
        memory_bound_time = memory_traffic / self.hardware.memory_bandwidth
        
        # Calculate compute time
        compute_time = total_flops / achievable_flops
        
        # Actual time is max of compute and memory bound
        predicted_time = max(compute_time, memory_bound_time)
        
        # Calculate metrics
        arithmetic_intensity = self.calculate_arithmetic_intensity(config)
        achieved_flops = total_flops / predicted_time
        efficiency = achieved_flops / peak_flops
        
        return {
            "predicted_time_seconds": predicted_time,
            "total_flops": total_flops,
            "peak_flops_per_second": peak_flops,
            "achieved_flops_per_second": achieved_flops,
            "tensor_core_utilization": utilization,
            "arithmetic_intensity": arithmetic_intensity,
            "efficiency_percent": efficiency * 100,
            "memory_traffic_bytes": memory_traffic,
            "compute_bound_time": compute_time,
            "memory_bound_time": memory_bound_time,
            "bottleneck": "compute" if compute_time > memory_bound_time else "memory"
        }
    
    def analyze_scaling(self, base_config: StencilConfig, scale_factors: List[float]) -> List[Dict[str, float]]:
        """Analyze performance scaling with problem size"""
        results = []
        
        for factor in scale_factors:
            scaled_config = StencilConfig(
                shape=base_config.shape,
                dimensions=tuple(int(d * factor) for d in base_config.dimensions),
                iterations=base_config.iterations,
                radius=base_config.radius,
                halo_size=base_config.halo_size,
                data_type=base_config.data_type
            )
            
            performance = self.predict_performance(scaled_config)
            performance["scale_factor"] = factor
            results.append(performance)
        
        return results