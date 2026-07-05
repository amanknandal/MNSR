import os
import json
import time
import asyncio
from pathlib import Path

# Framework imports
from evaluation.experiment import MNSRExperiment
from evaluation.benchmark import BenchmarkReport
from evaluation.statistical_tests import StatisticalAnalyzer
from evaluation.visualize import ResultsVisualizer
from evaluation.ablation import AblationStudy
from models.phi3 import Phi3Mini

# Configuration
DATASET_DIR = "datasets"
RESULTS_DIR = "results"
MAX_CONCURRENT_REQUESTS = 4 

def verify_and_get_datasets():
    """Checks the datasets directory and filters target JSON files."""
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Error: The directory '{DATASET_DIR}' was not found.")
        print("Please create it and place your 4 benchmark JSON files inside.")
        return []

    # Look specifically for your 4 core files
    target_files = ["gsm8k.json", "halueval_qa.json", "strategyqa.json", "truthful_qa.json"]
    found_datasets = []

    for file in target_files:
        path = os.path.join(DATASET_DIR, file)
        if os.path.exists(path):
            found_datasets.append(path)
        else:
            print(f"⚠️ Warning: Expected target file '{file}' is missing from '{DATASET_DIR}/'")

    # Fallback to any JSON file if specific targets aren't exactly named right
    if not found_datasets:
        found_datasets = [
            os.path.join(DATASET_DIR, f) 
            for f in os.listdir(DATASET_DIR) 
            if f.endswith(".json")
        ]

    return sorted(found_datasets)


def run_pipeline_for_dataset(dataset_path):
    """Executes the complete 5-step MNSR matrix evaluation on a single dataset file."""
    # Extract clean file name without extension
    dataset_name = Path(dataset_path).stem
    
    # Create isolated output directory for results
    output_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"🚀 MNSR EVALUATION STARTED FOR TRACK: {dataset_name.upper()}")
    print(f"⚙️ Hardware Accelerator Profile: Async Ollama (Throttle Limit: {MAX_CONCURRENT_REQUESTS})")
    print("=" * 80)

    # Performance logging
    start_time = time.time()

    # Define unique output file paths for this dataset loop
    results_file = os.path.join(output_dir, "results.json")
    benchmark_file = os.path.join(output_dir, "benchmark_report.json")
    ablation_file = os.path.join(output_dir, "ablation_results.json")

    # -------------------------------------------------------------
    # [Step 1/5] Core Comparative Experiment
    # -------------------------------------------------------------
    print(f"\n[Step 1/5] Running Core Experiments on {dataset_name}...")
    experiment = MNSRExperiment(dataset_path)
    
    # Dynamically pass down our hardware optimization parameters if supported
    if hasattr(experiment, 'model') and isinstance(experiment.model, Phi3Mini):
        experiment.model.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    experiment.run()
    experiment.save_results(results_file)
    experiment.print_summary()

    # -------------------------------------------------------------
    # [Step 2/5] Benchmark Matrix Compiler
    # -------------------------------------------------------------
    print("\n[Step 2/5] Compiling Benchmark Matrix Report...")
    bench = BenchmarkReport(results_file)
    bench.save(benchmark_file)
    bench.print_report()

    # -------------------------------------------------------------
    # [Step 3/5] Non-Parametric Significance Analytics
    # -------------------------------------------------------------
    print("\n[Step 3/5] Computing Non-Parametric Significance Statistics...")
    analyzer = StatisticalAnalyzer(results_file)
    analyzer.report()

    # -------------------------------------------------------------
    # [Step 4/5] Component Ablation Studies
    # -------------------------------------------------------------
    print("\n[Step 4/5] Executing Component Ablation Studies...")
    ablation = AblationStudy(dataset_path)
    ablation.run()
    ablation.save(ablation_file)
    ablation.print_report()

    # -------------------------------------------------------------
    # [Step 5/5] Publication Graph Rendering
    # -------------------------------------------------------------
    print("\n[Step 5/5] Exporting Paper Figures...")
    try:
        visualizer = ResultsVisualizer(
            benchmark_file=benchmark_file,
            ablation_file=ablation_file,
            output_dir=output_dir      # Cleanly saves charts to results/dataset_name/
        )
        visualizer.generate_all()
    except Exception as e:
        print(f"⚠️ Visualization step warning: {e}")
        print("Continuing pipeline execution...")

    elapsed_time = time.time() - start_time
    print(f"\n✅ Completed dataset run for '{dataset_name}' in {elapsed_time / 60:.2f} minutes.")


def main():
    print("\n" + "#" * 80)
    print("### INITIALIZING ACCELERATED LOCAL MNSR EXPERIMENT MATRIX RUNNER ###")
    print("#" * 80)

    # 1. Fetch valid dataset assets
    datasets = verify_and_get_datasets()
    if not datasets:
        print("❌ Pipeline execution halted: No dataset profiles found.")
        return

    # Count entries inside datasets to confirm correct sizing
    print(f"📋 Verification check: Found {len(datasets)} dataset queues to process.")
    for idx, path in enumerate(datasets, 1):
        try:
            with open(path, "r", encoding="utf-8") as f:
                samples = json.load(f)
                print(f"  [{idx}] {os.path.basename(path)} -> Detected {len(samples)} data rows.")
        except Exception:
            print(f"  [{idx}] {os.path.basename(path)} -> Unable to read format preview.")

    # 2. Iterate sequentially through each heavy task track
    total_start = time.time()
    for dataset in datasets:
        try:
            run_pipeline_for_dataset(dataset)
        except Exception as err:
            print(f"\n💥 CRITICAL: Track crashed on dataset '{os.path.basename(dataset)}' due to error: {err}")
            print("Skipping to next matrix file to preserve execution queue flow...\n")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("🌟 SUCCESS: ALL REASONING DATASETS EVALUATED BY THE ACCELERATED MNSR ENGINE.")
    print(f"⏱️ Total Wall-Clock Execution Time: {total_elapsed / 60:.2f} minutes.")
    print("📂 Review final compilation profiles and PNGs inside your 'results/' folder.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
