import json
from pathlib import Path
import matplotlib.pyplot as plt
class ResultsVisualizer:
    """
    Generates publication-ready figures for the MNSR evaluation paper.
    Adheres to clean academic typography and formatting styling rules.
    """
    def __init__(self,
                 benchmark_file="benchmark_report.json",
                 ablation_file="ablation_results.json"):
        self.benchmark = self._load(benchmark_file)
        self.ablation = self._load(ablation_file)
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.edgecolor'] = '#CCCCCC'
        plt.rcParams['axes.linewidth'] = 0.8
        self.colors = ['#5F7D95', '#2E4A62', '#A3B3C2', '#D1D9E0']
    def _load(self, filename: str):
        path = Path(filename)
        if not path.exists():
            print(f"Warning: Visualization source data file '{filename}' was not found.")
            return None
        with open(path, "r", encoding="utf-8") as f:
            try:
                return json.load(f)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
                return None
    def accuracy_comparison(self):
        if not self.benchmark:
            return
        labels = ["Baseline CoT", "MNSR (Ours)"]
        values = [
            self.benchmark.get("Baseline System Accuracy", 0.0) * 100,
            self.benchmark.get("MNSR System Accuracy", 0.0) * 100
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, values, color=[self.colors[0], self.colors[1]], width=0.5, edgecolor='#333333', linewidth=0.5)
        ax.set_ylabel("Task Accuracy (%)", fontsize=10, fontweight='bold', labelpad=8)
        ax.set_title("System-Level Accuracy Improvement Matrix", fontsize=11, fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig("figure_accuracy.png", dpi=300)
        plt.close()
    def latency_comparison(self):
        if not self.benchmark:
            return
        labels = ["Baseline CoT", "MNSR (Ours)"]
        values = [
            self.benchmark.get("Avg Baseline Latency (sec)", 0.0),
            self.benchmark.get("Avg MNSR Latency (sec)", 0.0)
        ]
        fig, ax = plt.subplots(figsize=(5, 4))
        bars = ax.bar(labels, values, color=[self.colors[0], self.colors[1]], width=0.5, edgecolor='#333333', linewidth=0.5)
    
        ax.set_ylabel("Execution Latency (seconds)", fontsize=10, fontweight='bold', labelpad=8)
        ax.set_title("Average Computation Overhead Latency", fontsize=11, fontweight='bold', pad=12)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        max_val = max(values) if values else 1.0
        ax.set_ylim(0, max_val * 1.2)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}s',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig("figure_latency.png", dpi=300)
        plt.close()
    def ablation_plot(self):
        if not self.ablation:
            return
        labels = list(self.ablation.keys())
        values = [self.ablation[k].get("mnsr_accuracy", 0.0) * 100 for k in labels]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        bars = ax.bar(labels, values, color=self.colors, width=0.5, edgecolor='#333333', linewidth=0.5)
        
        ax.set_ylabel("Task Accuracy (%)", fontsize=10, fontweight='bold', labelpad=8)
        ax.set_title("Component Ablation Architecture Degradation Analysis", fontsize=11, fontweight='bold', pad=12)
        ax.set_ylim(0, 105)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.set_axisbelow(True)
        plt.xticks(fontsize=9)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

        plt.tight_layout()
        plt.savefig("figure_ablation.png", dpi=300)
        plt.close()
    def generate_all(self):
        self.accuracy_comparison()
        self.latency_comparison()
        self.ablation_plot()
        print("\nAll academic publication-ready figures exported successfully.")