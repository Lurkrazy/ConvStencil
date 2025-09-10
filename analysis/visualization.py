"""
Visualization Tools for ConvStencil Performance Analysis

This module provides comprehensive visualization capabilities for 
performance analysis results.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class PerformanceVisualizer:
    """Visualization tools for ConvStencil performance analysis"""
    
    def __init__(self, output_dir: str = "plots"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure matplotlib for better plots
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_performance_comparison(self, results: Dict[str, List[Dict]], 
                                  save_name: str = "performance_comparison.png"):
        """Plot performance comparison between different methods"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ConvStencil Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract data for plotting
        methods = list(results.keys())
        
        # Throughput comparison
        ax = axes[0, 0]
        throughputs = {}
        for method, data in results.items():
            throughputs[method] = [d.get('throughput', 0) for d in data if isinstance(d, dict)]
        
        if throughputs:
            positions = np.arange(len(methods))
            for i, (method, values) in enumerate(throughputs.items()):
                if values:
                    ax.bar(i, np.mean(values), yerr=np.std(values), 
                          capsize=5, label=method, alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Throughput (GFLOPS)')
            ax.set_title('Throughput Comparison')
            ax.set_xticks(positions)
            ax.set_xticklabels(methods, rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Efficiency comparison
        ax = axes[0, 1]
        efficiencies = {}
        for method, data in results.items():
            efficiencies[method] = [d.get('efficiency', 0) for d in data if isinstance(d, dict)]
        
        if efficiencies:
            for i, (method, values) in enumerate(efficiencies.items()):
                if values:
                    ax.bar(i, np.mean(values), yerr=np.std(values), 
                          capsize=5, label=method, alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Efficiency (%)')
            ax.set_title('Efficiency Comparison')
            ax.set_xticks(positions)
            ax.set_xticklabels(methods, rotation=45)
            ax.grid(True, alpha=0.3)
        
        # Execution time comparison
        ax = axes[1, 0]
        exec_times = {}
        for method, data in results.items():
            exec_times[method] = [d.get('execution_time', 0) for d in data if isinstance(d, dict)]
        
        if exec_times:
            for i, (method, values) in enumerate(exec_times.items()):
                if values:
                    ax.bar(i, np.mean(values), yerr=np.std(values), 
                          capsize=5, label=method, alpha=0.8)
            
            ax.set_xlabel('Method')
            ax.set_ylabel('Execution Time (seconds)')
            ax.set_title('Execution Time Comparison')
            ax.set_xticks(positions)
            ax.set_xticklabels(methods, rotation=45)
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        # Speedup comparison (relative to first method)
        ax = axes[1, 1]
        if len(methods) > 1 and throughputs:
            baseline_method = methods[0]
            baseline_perf = np.mean(throughputs[baseline_method]) if throughputs[baseline_method] else 1
            
            speedups = []
            for method in methods:
                if throughputs[method]:
                    speedup = np.mean(throughputs[method]) / baseline_perf
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            
            bars = ax.bar(range(len(methods)), speedups, alpha=0.8)
            ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Baseline')
            
            ax.set_xlabel('Method')
            ax.set_ylabel(f'Speedup vs {baseline_method}')
            ax.set_title('Relative Speedup')
            ax.set_xticks(range(len(methods)))
            ax.set_xticklabels(methods, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for i, (bar, speedup) in enumerate(zip(bars, speedups)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'{speedup:.2f}x', ha='center', va='bottom')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance comparison saved to {save_path}")
    
    def plot_scaling_analysis(self, scaling_data: List[Dict], 
                             save_name: str = "scaling_analysis.png"):
        """Plot scaling behavior analysis"""
        
        if not scaling_data:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('ConvStencil Scaling Analysis', fontsize=16, fontweight='bold')
        
        # Extract data
        scale_factors = [d.get('scale_factor', 1) for d in scaling_data]
        throughputs = [d.get('achieved_flops_per_second', 0) / 1e9 for d in scaling_data]
        efficiencies = [d.get('efficiency_percent', 0) for d in scaling_data]
        mem_traffic = [d.get('memory_traffic_bytes', 0) / 1e9 for d in scaling_data]
        
        # Problem sizes
        problem_sizes = []
        for d in scaling_data:
            if 'total_flops' in d and 'iterations' in d:
                # Estimate problem size from FLOPS
                problem_sizes.append(d['total_flops'] / d.get('iterations', 1))
            else:
                problem_sizes.append(0)
        
        # Throughput vs scale factor
        ax = axes[0, 0]
        ax.plot(scale_factors, throughputs, 'o-', linewidth=2, markersize=8)
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Throughput (GFLOPS)')
        ax.set_title('Throughput Scaling')
        ax.grid(True, alpha=0.3)
        
        # Efficiency vs scale factor
        ax = axes[0, 1]
        ax.plot(scale_factors, efficiencies, 'o-', linewidth=2, markersize=8, color='red')
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Efficiency (%)')
        ax.set_title('Efficiency Scaling')
        ax.grid(True, alpha=0.3)
        
        # Memory traffic scaling
        ax = axes[0, 2]
        ax.plot(scale_factors, mem_traffic, 'o-', linewidth=2, markersize=8, color='green')
        ax.set_xlabel('Scale Factor')
        ax.set_ylabel('Memory Traffic (GB)')
        ax.set_title('Memory Traffic Scaling')
        ax.grid(True, alpha=0.3)
        
        # Log-log plots for better understanding of scaling behavior
        if any(problem_sizes):
            # Throughput vs problem size
            ax = axes[1, 0]
            valid_data = [(ps, tp) for ps, tp in zip(problem_sizes, throughputs) if ps > 0 and tp > 0]
            if valid_data:
                ps_valid, tp_valid = zip(*valid_data)
                ax.loglog(ps_valid, tp_valid, 'o-', linewidth=2, markersize=8)
                ax.set_xlabel('Problem Size (FLOPS)')
                ax.set_ylabel('Throughput (GFLOPS)')
                ax.set_title('Throughput vs Problem Size')
                ax.grid(True, alpha=0.3)
        
        # Efficiency distribution
        ax = axes[1, 1]
        ax.hist(efficiencies, bins=10, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Efficiency (%)')
        ax.set_ylabel('Frequency')
        ax.set_title('Efficiency Distribution')
        ax.grid(True, alpha=0.3)
        
        # Scaling efficiency (ideal vs actual)
        ax = axes[1, 2]
        if len(scale_factors) > 1 and throughputs:
            ideal_scaling = [throughputs[0] * sf for sf in scale_factors]
            ax.plot(scale_factors, ideal_scaling, '--', linewidth=2, 
                   alpha=0.7, label='Ideal Scaling')
            ax.plot(scale_factors, throughputs, 'o-', linewidth=2, 
                   markersize=8, label='Actual Performance')
            ax.set_xlabel('Scale Factor')
            ax.set_ylabel('Throughput (GFLOPS)')
            ax.set_title('Scaling Efficiency')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scaling analysis saved to {save_path}")
    
    def plot_memory_analysis(self, memory_data: Dict, 
                           save_name: str = "memory_analysis.png"):
        """Plot memory performance analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ConvStencil Memory Analysis', fontsize=16, fontweight='bold')
        
        # Cache behavior
        if 'cache_behavior' in memory_data:
            cache_data = memory_data['cache_behavior']
            ax = axes[0, 0]
            
            metrics = ['l1_hit_rate', 'l2_hit_rate', 'spatial_locality', 'temporal_locality']
            values = [cache_data.get(metric, 0) * 100 for metric in metrics]
            labels = ['L1 Hit Rate', 'L2 Hit Rate', 'Spatial Locality', 'Temporal Locality']
            
            bars = ax.bar(labels, values, alpha=0.8)
            ax.set_ylabel('Percentage (%)')
            ax.set_title('Cache Behavior Analysis')
            ax.set_ylim(0, 100)
            plt.setp(ax.get_xticklabels(), rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                       f'{value:.1f}%', ha='center', va='bottom')
        
        # Bandwidth utilization
        if 'bandwidth_utilization' in memory_data:
            bw_data = memory_data['bandwidth_utilization']
            ax = axes[0, 1]
            
            theoretical = bw_data.get('theoretical_bandwidth', 0) / 1e9
            achieved = bw_data.get('achieved_bandwidth', 0) / 1e9
            utilization = bw_data.get('bandwidth_utilization', 0) * 100
            
            bars = ax.bar(['Theoretical', 'Achieved'], [theoretical, achieved], alpha=0.8)
            ax.set_ylabel('Bandwidth (GB/s)')
            ax.set_title(f'Memory Bandwidth\n(Utilization: {utilization:.1f}%)')
            
            # Add value labels
            for bar, value in zip(bars, [theoretical, achieved]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                       f'{value:.0f} GB/s', ha='center', va='bottom')
        
        # Memory footprint breakdown
        if 'memory_footprint' in memory_data:
            footprint = memory_data['memory_footprint']
            ax = axes[1, 0]
            
            components = ['input_size_bytes', 'output_size_bytes', 'transformation_buffer_bytes']
            values = [footprint.get(comp, 0) / 1e6 for comp in components]  # Convert to MB
            labels = ['Input Data', 'Output Data', 'Transformation Buffer']
            
            # Remove zero values
            non_zero = [(l, v) for l, v in zip(labels, values) if v > 0]
            if non_zero:
                labels, values = zip(*non_zero)
                ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
                ax.set_title('Memory Footprint Breakdown')
        
        # Memory hierarchy performance
        ax = axes[1, 1]
        if 'dram_time_seconds' in memory_data and 'l2_time_seconds' in memory_data:
            dram_time = memory_data['dram_time_seconds']
            l2_time = memory_data['l2_time_seconds']
            total_time = memory_data.get('memory_bound_time_seconds', dram_time + l2_time)
            
            times = [dram_time, l2_time, total_time - dram_time - l2_time]
            labels = ['DRAM Access', 'L2 Cache', 'Other']
            colors = ['red', 'orange', 'lightblue']
            
            # Remove negative or zero values
            valid_data = [(l, t, c) for l, t, c in zip(labels, times, colors) if t > 0]
            if valid_data:
                labels, times, colors = zip(*valid_data)
                ax.pie(times, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
                ax.set_title('Memory Access Time Breakdown')
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Memory analysis saved to {save_path}")
    
    def plot_tensor_core_analysis(self, tensor_data: Dict, 
                                 save_name: str = "tensor_core_analysis.png"):
        """Plot tensor core utilization analysis"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('ConvStencil Tensor Core Analysis', fontsize=16, fontweight='bold')
        
        # Efficiency breakdown
        if 'efficiency_breakdown' in tensor_data:
            breakdown = tensor_data['efficiency_breakdown']
            ax = axes[0, 0]
            
            metrics = []
            values = []
            for key, value in breakdown.items():
                if isinstance(value, (int, float)) and key.endswith('_efficiency'):
                    metrics.append(key.replace('_efficiency', '').replace('_', ' ').title())
                    values.append(value * 100)
            
            if metrics:
                bars = ax.bar(metrics, values, alpha=0.8)
                ax.set_ylabel('Efficiency (%)')
                ax.set_title('Tensor Core Efficiency Breakdown')
                ax.set_ylim(0, 100)
                plt.setp(ax.get_xticklabels(), rotation=45)
                
                # Add value labels
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                           f'{value:.1f}%', ha='center', va='bottom')
        
        # Utilization vs Performance
        ax = axes[0, 1]
        utilization = tensor_data.get('tensor_core_utilization', 0) * 100
        achieved_tflops = tensor_data.get('achieved_tflops', 0)
        
        # Create a gauge-like plot
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax_polar = fig.add_subplot(2, 2, 2, projection='polar')
        ax_polar.plot(theta, r, 'k-', linewidth=3)
        ax_polar.fill_between(theta, 0, r, alpha=0.1)
        
        # Add utilization indicator
        util_angle = np.pi * (1 - utilization / 100)
        ax_polar.plot([util_angle, util_angle], [0, 1], 'r-', linewidth=5)
        ax_polar.text(np.pi/2, 0.5, f'{utilization:.1f}%\n{achieved_tflops:.2f} TFLOPS', 
                     ha='center', va='center', fontsize=12, fontweight='bold')
        ax_polar.set_ylim(0, 1)
        ax_polar.set_theta_zero_location('W')
        ax_polar.set_theta_direction(1)
        ax_polar.set_title('Tensor Core Utilization')
        ax_polar.set_rticks([])
        
        # Remove the original subplot
        axes[0, 1].remove()
        
        # Performance comparison
        ax = axes[1, 0]
        theoretical_peak = 312  # A100 theoretical peak TFLOPS for tensor cores
        achieved = achieved_tflops
        efficiency = tensor_data.get('tensor_core_efficiency', 0) * 100
        
        categories = ['Theoretical Peak', 'Achieved Performance']
        values = [theoretical_peak, achieved]
        colors = ['lightblue', 'darkblue']
        
        bars = ax.bar(categories, values, color=colors, alpha=0.8)
        ax.set_ylabel('Performance (TFLOPS)')
        ax.set_title(f'Tensor Core Performance\n(Efficiency: {efficiency:.1f}%)')
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                   f'{value:.1f}', ha='center', va='bottom')
        
        # Tensor operations analysis
        ax = axes[1, 1]
        if 'tensor_core_ops_required' in tensor_data:
            ops_required = tensor_data['tensor_core_ops_required']
            execution_time = tensor_data.get('execution_time_seconds', 1)
            
            # Create timeline visualization
            time_points = np.linspace(0, execution_time, 100)
            ops_rate = ops_required / execution_time
            cumulative_ops = time_points * ops_rate
            
            ax.plot(time_points, cumulative_ops / 1e6, linewidth=2)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Cumulative Operations (Million)')
            ax.set_title('Tensor Core Operation Timeline')
            ax.grid(True, alpha=0.3)
            
            # Add final stats
            ax.text(0.02, 0.98, f'Total Ops: {ops_required/1e6:.1f}M\nRate: {ops_rate/1e6:.1f}M/s',
                   transform=ax.transAxes, va='top', ha='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Tensor core analysis saved to {save_path}")
    
    def create_summary_dashboard(self, all_results: Dict, 
                               save_name: str = "performance_dashboard.png"):
        """Create a comprehensive performance dashboard"""
        
        fig = plt.figure(figsize=(20, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        fig.suptitle('ConvStencil Performance Dashboard', fontsize=20, fontweight='bold')
        
        # Extract key metrics from results
        if 'performance_prediction' in all_results:
            perf = all_results['performance_prediction']
            
            # Overall performance gauge
            ax = fig.add_subplot(gs[0, 0])
            efficiency = perf.get('efficiency_percent', 0)
            self._create_gauge(ax, efficiency, 'Overall Efficiency (%)', 0, 100)
            
            # Throughput
            ax = fig.add_subplot(gs[0, 1])
            throughput = perf.get('achieved_flops_per_second', 0) / 1e9
            peak_throughput = perf.get('peak_flops_per_second', 1) / 1e9
            ax.bar(['Achieved', 'Peak'], [throughput, peak_throughput], 
                  color=['darkgreen', 'lightgray'], alpha=0.8)
            ax.set_ylabel('GFLOPS')
            ax.set_title('Throughput Comparison')
            
            # Bottleneck analysis
            ax = fig.add_subplot(gs[0, 2])
            compute_time = perf.get('compute_bound_time', 0)
            memory_time = perf.get('memory_bound_time', 0)
            bottleneck = perf.get('bottleneck', 'unknown')
            
            if compute_time > 0 and memory_time > 0:
                times = [compute_time, memory_time]
                labels = ['Compute', 'Memory']
                colors = ['blue', 'red']
                ax.pie(times, labels=labels, colors=colors, autopct='%1.1f%%')
                ax.set_title(f'Bottleneck: {bottleneck.title()}')
        
        # Add more visualizations as space allows...
        
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance dashboard saved to {save_path}")
    
    def _create_gauge(self, ax, value, title, min_val, max_val):
        """Create a gauge-style visualization"""
        theta = np.linspace(0, np.pi, 100)
        r = np.ones_like(theta)
        
        ax = plt.subplot(projection='polar')
        ax.plot(theta, r, 'k-', linewidth=3)
        ax.fill_between(theta, 0, r, alpha=0.1)
        
        # Value indicator
        val_angle = np.pi * (1 - (value - min_val) / (max_val - min_val))
        ax.plot([val_angle, val_angle], [0, 1], 'r-', linewidth=5)
        ax.text(np.pi/2, 0.5, f'{value:.1f}', ha='center', va='center', 
               fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, 1)
        ax.set_theta_zero_location('W')
        ax.set_theta_direction(1)
        ax.set_title(title)
        ax.set_rticks([])