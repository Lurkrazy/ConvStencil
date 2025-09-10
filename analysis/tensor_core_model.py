"""
Tensor Core Specific Performance Model

This module provides detailed modeling for Tensor Core operations
in the context of ConvStencil transformations.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from performance_model import StencilConfig, HardwareSpec, TensorCoreConfig

@dataclass
class TensorCoreOperation:
    """Represents a tensor core operation"""
    m: int  # Matrix dimensions
    n: int
    k: int
    data_type: str
    operation_count: int

class TensorCoreModel:
    """Detailed tensor core performance modeling"""
    
    def __init__(self, hardware: HardwareSpec, tensor_config: TensorCoreConfig):
        self.hardware = hardware
        self.tensor_config = tensor_config
        
        # Tensor core throughput specifications (operations per clock)
        self.tensor_core_throughput = {
            "8.0": {  # A100
                "double": {"m8n8k4": 1},
                "float": {"m16n16k8": 1}
            },
            "7.5": {  # T4
                "float": {"m16n16k8": 1},
                "half": {"m16n16k16": 1}
            }
        }
    
    def get_tensor_core_specs(self, data_type: str) -> Dict[str, int]:
        """Get tensor core specifications for given data type"""
        cc = self.hardware.compute_capability
        if cc in self.tensor_core_throughput:
            if data_type in self.tensor_core_throughput[cc]:
                return self.tensor_core_throughput[cc][data_type]
        
        # Default to double precision A100 specs
        return {"m8n8k4": 1}
    
    def calculate_stencil2row_efficiency(self, config: StencilConfig) -> Dict[str, float]:
        """Calculate efficiency of stencil2row transformation for tensor cores"""
        
        # Analyze how stencil points map to matrix multiplication
        if config.shape.value.startswith("star_1d"):
            points = 3 if "1r" in config.shape.value else 5
            matrix_efficiency = self._analyze_1d_mapping(points, config.dimensions[0])
        elif config.shape.value.startswith("box_2d") or config.shape.value.startswith("star_2d"):
            if "1r" in config.shape.value:
                points = 9 if "box" in config.shape.value else 5
            else:  # 3r
                points = 49 if "box" in config.shape.value else 13
            matrix_efficiency = self._analyze_2d_mapping(points, config.dimensions)
        else:  # 3D
            points = 27 if "box" in config.shape.value else 7
            matrix_efficiency = self._analyze_3d_mapping(points, config.dimensions)
        
        return matrix_efficiency
    
    def _analyze_1d_mapping(self, points: int, size: int) -> Dict[str, float]:
        """Analyze 1D stencil mapping to matrix operations"""
        # For 1D stencils, we create matrices where each row represents output points
        # and columns represent stencil coefficients
        
        # Tensor core works best with 8x8 matrices
        rows_per_tile = 8
        cols_per_tile = 8
        
        # Calculate how many complete tiles we can form
        total_outputs = size
        complete_row_tiles = total_outputs // rows_per_tile
        remaining_rows = total_outputs % rows_per_tile
        
        # For columns, we need to pad stencil points to fit tensor core dimensions
        padded_points = ((points + cols_per_tile - 1) // cols_per_tile) * cols_per_tile
        padding_overhead = (padded_points - points) / padded_points
        
        # Efficiency based on tile utilization
        row_efficiency = complete_row_tiles * rows_per_tile / total_outputs
        if remaining_rows > 0:
            row_efficiency += (remaining_rows / rows_per_tile) * (remaining_rows / total_outputs)
        
        col_efficiency = 1.0 - padding_overhead
        
        return {
            "row_efficiency": row_efficiency,
            "col_efficiency": col_efficiency, 
            "overall_efficiency": row_efficiency * col_efficiency,
            "padding_overhead": padding_overhead
        }
    
    def _analyze_2d_mapping(self, points: int, dimensions: Tuple[int, int]) -> Dict[str, float]:
        """Analyze 2D stencil mapping to matrix operations"""
        h, w = dimensions
        total_outputs = h * w
        
        # For 2D, we can use different blocking strategies
        # Strategy 1: Block-wise processing
        block_size = 32  # Typical block size for 2D stencils
        blocks_h = (h + block_size - 1) // block_size
        blocks_w = (w + block_size - 1) // block_size
        
        # Calculate tensor core utilization within each block
        points_per_block = min(block_size * block_size, total_outputs)
        tensor_operations_per_block = (points_per_block * points) // (8 * 8 * 4)  # 8x8x4 tensor core
        
        # Efficiency factors
        spatial_locality = min(1.0, (block_size * block_size) / (h * w))
        coefficient_reuse = min(1.0, points / 8.0)  # How well stencil points fit in tensor core
        
        # Memory access pattern efficiency
        memory_efficiency = self._calculate_memory_access_efficiency(dimensions, points)
        
        return {
            "spatial_locality": spatial_locality,
            "coefficient_reuse": coefficient_reuse,
            "memory_efficiency": memory_efficiency,
            "overall_efficiency": spatial_locality * coefficient_reuse * memory_efficiency,
            "tensor_ops_per_block": tensor_operations_per_block
        }
    
    def _analyze_3d_mapping(self, points: int, dimensions: Tuple[int, int, int]) -> Dict[str, float]:
        """Analyze 3D stencil mapping to matrix operations"""
        d, h, w = dimensions
        total_outputs = d * h * w
        
        # 3D stencils are more complex to map efficiently
        # We typically use a combination of 2D slices and depth-wise operations
        
        # Analyze slice-wise efficiency (treating each depth slice as 2D)
        slice_efficiency = self._analyze_2d_mapping(points, (h, w))
        
        # Depth dimension adds complexity
        depth_factor = min(1.0, d / 8.0)  # Efficiency decreases with very large depth
        
        # 3D stencils often have lower cache efficiency
        cache_efficiency = max(0.6, 1.0 - (total_outputs * 8) / self.hardware.l2_cache_size)
        
        return {
            "slice_efficiency": slice_efficiency["overall_efficiency"],
            "depth_factor": depth_factor,
            "cache_efficiency": cache_efficiency,
            "overall_efficiency": slice_efficiency["overall_efficiency"] * depth_factor * cache_efficiency
        }
    
    def _calculate_memory_access_efficiency(self, dimensions: Tuple[int, ...], points: int) -> float:
        """Calculate memory access pattern efficiency"""
        # Factors affecting memory efficiency:
        # 1. Coalesced access patterns
        # 2. Cache utilization
        # 3. Bandwidth utilization
        
        if len(dimensions) == 1:
            # 1D has good spatial locality
            return 0.9
        elif len(dimensions) == 2:
            h, w = dimensions
            # 2D efficiency depends on how well we can coalesce accesses
            # Wider matrices generally have better coalescing
            coalescing_factor = min(1.0, w / 128.0)  # 128 is typical coalescing width
            return 0.7 + 0.2 * coalescing_factor
        else:
            # 3D has more complex access patterns
            d, h, w = dimensions
            # Efficiency decreases with depth due to more complex access patterns
            return max(0.5, 0.8 - (d / 100.0) * 0.2)
    
    def predict_tensor_core_performance(self, config: StencilConfig) -> Dict[str, float]:
        """Predict tensor core specific performance metrics"""
        
        # Get transformation efficiency
        efficiency = self.calculate_stencil2row_efficiency(config)
        
        # Calculate tensor core utilization
        total_elements = np.prod(config.dimensions)
        stencil_props = self._get_stencil_properties(config.shape.value)
        
        # Estimate number of tensor core operations required
        total_matrix_ops = total_elements * stencil_props["points"]
        tensor_core_ops_required = total_matrix_ops // (8 * 8 * 4)  # Operations per tensor core
        
        # Calculate throughput
        tensor_cores_available = self.hardware.tensor_core_count
        ops_per_second = tensor_cores_available * self.hardware.tensor_core_frequency
        
        # Apply efficiency factors
        effective_ops_per_second = ops_per_second * efficiency["overall_efficiency"]
        
        # Predict execution time
        execution_time = tensor_core_ops_required / effective_ops_per_second
        
        # Calculate achieved performance
        achieved_tflops = (total_matrix_ops * config.iterations) / (execution_time * 1e12)
        
        return {
            "execution_time_seconds": execution_time,
            "tensor_core_ops_required": tensor_core_ops_required,
            "tensor_core_efficiency": efficiency["overall_efficiency"],
            "achieved_tflops": achieved_tflops,
            "tensor_core_utilization": min(1.0, tensor_core_ops_required / tensor_cores_available),
            "efficiency_breakdown": efficiency
        }
    
    def _get_stencil_properties(self, shape_str: str) -> Dict[str, int]:
        """Helper to get stencil properties from string"""
        properties = {
            "star_1d1r": {"points": 3, "radius": 1},
            "star_1d2r": {"points": 5, "radius": 2},
            "box_2d1r": {"points": 9, "radius": 1},
            "star_2d1r": {"points": 5, "radius": 1},
            "box_2d3r": {"points": 49, "radius": 3},
            "star_2d3r": {"points": 13, "radius": 3},
            "box_3d1r": {"points": 27, "radius": 1},
            "star_3d1r": {"points": 7, "radius": 1},
        }
        return properties.get(shape_str, {"points": 1, "radius": 1})
    
    def analyze_dual_tessellation(self, config: StencilConfig, 
                                tessellation_factors: List[Tuple[int, int]]) -> List[Dict[str, float]]:
        """Analyze performance with different dual tessellation strategies"""
        results = []
        
        for factor in tessellation_factors:
            if len(config.dimensions) >= 2:
                tile_h, tile_w = factor
                
                # Calculate number of tiles
                h, w = config.dimensions[:2]
                num_tiles_h = (h + tile_h - 1) // tile_h
                num_tiles_w = (w + tile_w - 1) // tile_w
                
                # Efficiency factors for this tessellation
                tile_utilization = (h * w) / (num_tiles_h * tile_h * num_tiles_w * tile_w)
                
                # Tensor core efficiency within tiles
                tile_tensor_efficiency = min(1.0, (tile_h * tile_w) / (8 * 8))
                
                # Memory efficiency (smaller tiles = better cache behavior)
                cache_efficiency = max(0.5, 1.0 - (tile_h * tile_w * 8) / self.hardware.shared_memory_size)
                
                overall_efficiency = tile_utilization * tile_tensor_efficiency * cache_efficiency
                
                result = {
                    "tessellation_factor": factor,
                    "tile_utilization": tile_utilization,
                    "tensor_core_efficiency": tile_tensor_efficiency,
                    "cache_efficiency": cache_efficiency,
                    "overall_efficiency": overall_efficiency,
                    "num_tiles": num_tiles_h * num_tiles_w
                }
                results.append(result)
        
        return results