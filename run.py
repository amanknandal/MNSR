import os
from evaluation.experiment import MNSRExperiment
from evaluation.benchmark import BenchmarkReport
from evaluation.statistical_tests import StatisticalAnalyzer
from evaluation.visualize import ResultsVisualizer
from evaluation.ablation import AblationStudy

def main():
    # 1. Define dataset path (Replace with your actual dataset file)
    dataset_path = "data/math_dataset.json" 
    
    # Check if a dummy dataset is needed for testing
    if not os.path.exists(dataset_path):
        os.makedirs("data", exist_ok=True)
        import json
        dummy_data = [
            {"question": "If John has 5 apples and buys 3 more, how many does he have?", "answer": "8"},
            {"question": "What is 12 multiplied by 11?", "answer": "132"}
        ]
        with open(dataset_path, "w", encoding="utf-8") as f:
            json.dump(dummy_data, f, indent=4)
        print(f"Created a dummy dataset at: {dataset_path}")

    print("\nStarting complete MNSR Evaluation Pipeline...")

    # 2. Run Main Comparative Experiment (Generates results.json)
    print("\n[Step 1/5] Running Core Experiments...")
    experiment = MNSRExperiment(dataset_path)
    experiment.run()
    experiment.save_results("results.json")
    experiment.print_summary()

    # 3. Generate Main Metric Report (Generates benchmark_report.json)
    print("\n[Step 2/5] Compiling Benchmark Matrix Report...")
    bench = BenchmarkReport("results.json")
    bench.save("benchmark_report.json")
    bench.print_report()

    # 4. Compute Statistical Significance
    print("\n[Step 3/5] Computing Non-Parametric Significance Statistics...")
    analyzer = StatisticalAnalyzer("results.json")
    analyzer.report()

    # 5. Execute Ablation Runs (Generates ablation_results.json)
    print("\n[Step 4/5] Executing Component Ablation Studies...")
    ablation = AblationStudy(dataset_path)
    ablation.run()
    ablation.save("ablation_results.json")
    ablation.print_report()

    # 6. Build Publication Plots (Generates figure_*.png files)
    print("\n[Step 5/5] Exporting Paper Figures...")
    visualizer = ResultsVisualizer(
        benchmark_file="benchmark_report.json",
        ablation_file="ablation_results.json"
    )
    visualizer.generate_all()

    print("\nExecution complete. All JSON records and PNG figures are ready.")

if __name__ == "__main__":
    main()