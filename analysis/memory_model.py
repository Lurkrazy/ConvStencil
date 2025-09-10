"""
Memory Performance Model

This module models memory bandwidth, latency, and access patterns
for ConvStencil operations.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from performance_model import StencilConfig, HardwareSpec

@dataclass
class MemoryHierarchy:
    """Memory hierarchy specifications"""
    l1_cache_size: int = 128 * 1024  # L1 data cache per SM
    l2_cache_size: int = 40 * 1024 * 1024  # L2 cache shared
    shared_memory_size: int = 164 * 1024  # Shared memory per SM
    register_file_size: int = 256 * 1024  # Register file per SM
    
    l1_latency: int = 1  # cycles
    l2_latency: int = 30  # cycles
    dram_latency: int = 300  # cycles
    
    l1_bandwidth: float = 19e12  # bytes/s per SM
    l2_bandwidth: float = 3.1e12  # bytes/s
    dram_bandwidth: float = 1555e9  # bytes/s

class MemoryModel:
    """Memory performance modeling for stencil computations"""
    
    def __init__(self, hardware: HardwareSpec):
        self.hardware = hardware
        self.memory_hierarchy = MemoryHierarchy()
        self.sm_count = 108  # A100 has 108 SMs
    
    def calculate_memory_footprint(self, config: StencilConfig) -> Dict[str, float]:
        """Calculate memory footprint for different levels of memory hierarchy"""
        data_size = 8 if config.data_type == "double" else 4  # bytes per element
        
        # Calculate total data size including halos
        if len(config.dimensions) == 1:
            input_size = (config.dimensions[0] + 2 * config.radius) * data_size
            output_size = config.dimensions[0] * data_size
            working_set = input_size + output_size
        elif len(config.dimensions) == 2:
            h, w = config.dimensions
            input_size = (h + 2 * config.radius) * (w + 2 * config.radius) * data_size
            output_size = h * w * data_size
            # Additional memory for stencil2row transformation
            transformation_buffer = input_size * 1.5  # 50% overhead for transformation
            working_set = input_size + output_size + transformation_buffer
        else:  # 3D
            d, h, w = config.dimensions
            input_size = (d + 2 * config.radius) * (h + 2 * config.radius) * (w + 2 * config.radius) * data_size
            output_size = d * h * w * data_size
            transformation_buffer = input_size * 2.0  # Higher overhead for 3D
            working_set = input_size + output_size + transformation_buffer
        
        return {
            "input_size_bytes": input_size,
            "output_size_bytes": output_size,
            "working_set_bytes": working_set,
            "transformation_buffer_bytes": transformation_buffer if len(config.dimensions) > 1 else 0,
            "total_memory_per_iteration": working_set
        }
    
    def analyze_cache_behavior(self, config: StencilConfig) -> Dict[str, float]:
        """Analyze cache hit rates and memory access patterns"""
        footprint = self.calculate_memory_footprint(config)
        
        # L1 cache analysis (per SM)
        working_set_per_sm = footprint["working_set_bytes"] / self.sm_count
        l1_hit_rate = min(1.0, self.memory_hierarchy.l1_cache_size / working_set_per_sm)
        
        # L2 cache analysis (shared across SMs)
        l2_hit_rate = min(1.0, self.memory_hierarchy.l2_cache_size / footprint["working_set_bytes"])
        
        # Spatial locality analysis
        spatial_locality = self._calculate_spatial_locality(config)
        
        # Temporal locality analysis  
        temporal_locality = self._calculate_temporal_locality(config)
        
        # Effective cache hit rates considering locality
        effective_l1_hit_rate = l1_hit_rate * spatial_locality * temporal_locality
        effective_l2_hit_rate = l2_hit_rate * spatial_locality
        
        return {
            "l1_hit_rate": effective_l1_hit_rate,
            "l2_hit_rate": effective_l2_hit_rate,
            "spatial_locality": spatial_locality,
            "temporal_locality": temporal_locality,
            "working_set_per_sm": working_set_per_sm
        }
    
    def _calculate_spatial_locality(self, config: StencilConfig) -> float:
        """Calculate spatial locality factor based on access patterns"""
        if len(config.dimensions) == 1:
            # 1D stencils have excellent spatial locality
            return 0.95
        elif len(config.dimensions) == 2:
            h, w = config.dimensions
            # 2D stencils - locality depends on access pattern
            # Row-major access has good locality for width dimension
            cache_line_elements = 128 // (8 if config.data_type == "double" else 4)  # elements per cache line
            width_locality = min(1.0, w / cache_line_elements)
            return 0.7 + 0.2 * width_locality
        else:
            # 3D stencils have more complex access patterns
            return 0.6
    
    def _calculate_temporal_locality(self, config: StencilConfig) -> float:
        """Calculate temporal locality factor"""
        # Temporal locality depends on how often data is reused
        # Stencil computations have good temporal locality due to overlapping neighborhoods
        
        stencil_radius = config.radius
        reuse_factor = min(1.0, (2 * stencil_radius + 1) / 8.0)  # Normalized reuse
        
        # Multiple iterations increase temporal locality
        iteration_factor = min(1.0, config.iterations / 10.0)
        
        return 0.5 + 0.3 * reuse_factor + 0.2 * iteration_factor
    
    def calculate_memory_bandwidth_utilization(self, config: StencilConfig) -> Dict[str, float]:
        """Calculate memory bandwidth utilization"""
        footprint = self.calculate_memory_footprint(config)
        cache_behavior = self.analyze_cache_behavior(config)
        
        # Calculate effective memory traffic
        total_accesses = footprint["working_set_bytes"] * config.iterations
        
        # Apply cache hit rates to determine actual DRAM traffic
        l1_miss_traffic = total_accesses * (1 - cache_behavior["l1_hit_rate"])
        l2_miss_traffic = l1_miss_traffic * (1 - cache_behavior["l2_hit_rate"])
        dram_traffic = l2_miss_traffic
        
        # Calculate access pattern efficiency
        access_efficiency = self._calculate_access_pattern_efficiency(config)
        
        # Effective bandwidth utilization
        theoretical_bandwidth = self.hardware.memory_bandwidth
        achieved_bandwidth = theoretical_bandwidth * access_efficiency
        
        return {
            "dram_traffic_bytes": dram_traffic,
            "l2_traffic_bytes": l1_miss_traffic,
            "access_pattern_efficiency": access_efficiency,
            "theoretical_bandwidth": theoretical_bandwidth,
            "achieved_bandwidth": achieved_bandwidth,
            "bandwidth_utilization": achieved_bandwidth / theoretical_bandwidth
        }
    
    def _calculate_access_pattern_efficiency(self, config: StencilConfig) -> float:
        """Calculate efficiency of memory access patterns"""
        base_efficiency = 0.8  # Base efficiency for well-optimized code
        
        # Coalescing efficiency
        coalescing_efficiency = self._calculate_coalescing_efficiency(config)
        
        # Bank conflict efficiency (for shared memory)
        bank_conflict_efficiency = self._calculate_bank_conflict_efficiency(config)
        
        return base_efficiency * coalescing_efficiency * bank_conflict_efficiency
    
    def _calculate_coalescing_efficiency(self, config: StencilConfig) -> float:
        """Calculate memory coalescing efficiency"""
        if len(config.dimensions) == 1:
            # 1D access patterns are naturally coalesced
            return 0.95
        elif len(config.dimensions) == 2:
            h, w = config.dimensions
            # Coalescing efficiency depends on access stride
            # Row-major access to width dimension is well coalesced
            warp_size = 32
            element_size = 8 if config.data_type == "double" else 4
            coalesced_elements = min(w, warp_size)
            efficiency = coalesced_elements / warp_size
            return max(0.5, efficiency)
        else:
            # 3D access patterns are more complex
            return 0.7
    
    def _calculate_bank_conflict_efficiency(self, config: StencilConfig) -> float:
        """Calculate shared memory bank conflict efficiency"""
        # A100 has 32 banks with 4-byte word size
        num_banks = 32
        bank_width = 4  # bytes
        
        element_size = 8 if config.data_type == "double" else 4
        
        if element_size > bank_width:
            # Double precision causes more bank conflicts
            return 0.8
        else:
            # Single precision has fewer conflicts
            return 0.95
    
    def predict_memory_performance(self, config: StencilConfig) -> Dict[str, float]:
        """Predict overall memory performance"""
        footprint = self.calculate_memory_footprint(config)
        cache_behavior = self.analyze_cache_behavior(config)
        bandwidth_util = self.calculate_memory_bandwidth_utilization(config)
        
        # Calculate memory-bound execution time
        dram_time = bandwidth_util["dram_traffic_bytes"] / bandwidth_util["achieved_bandwidth"]
        l2_time = bandwidth_util["l2_traffic_bytes"] / self.memory_hierarchy.l2_bandwidth
        
        # Total memory time includes cache hierarchy
        total_memory_time = dram_time + l2_time * 0.1  # L2 contributes less to total time
        
        # Memory efficiency metrics
        memory_efficiency = bandwidth_util["bandwidth_utilization"] * cache_behavior["l1_hit_rate"]
        
        return {
            "memory_bound_time_seconds": total_memory_time,
            "dram_time_seconds": dram_time,
            "l2_time_seconds": l2_time,
            "memory_efficiency": memory_efficiency,
            "cache_behavior": cache_behavior,
            "bandwidth_utilization": bandwidth_util,
            "memory_footprint": footprint
        }
    
    def analyze_data_layout_impact(self, config: StencilConfig, 
                                 layouts: List[str] = ["row_major", "column_major", "blocked"]) -> Dict[str, Dict[str, float]]:
        """Analyze impact of different data layouts on memory performance"""
        results = {}
        
        for layout in layouts:
            # Modify config for different layouts
            layout_config = self._apply_layout_config(config, layout)
            
            # Calculate performance for this layout
            perf = self.predict_memory_performance(layout_config)
            
            # Add layout-specific metrics
            if layout == "row_major":
                perf["layout_efficiency"] = 1.0  # Baseline
            elif layout == "column_major":
                perf["layout_efficiency"] = 0.7  # Generally worse for stencils
            elif layout == "blocked":
                perf["layout_efficiency"] = 1.1  # Can be better for cache behavior
            
            results[layout] = perf
        
        return results
    
    def _apply_layout_config(self, config: StencilConfig, layout: str) -> StencilConfig:
        """Apply layout-specific configuration adjustments"""
        # This is a simplified approach - in practice, layout affects
        # access patterns and cache behavior significantly
        return config  # For now, return unchanged