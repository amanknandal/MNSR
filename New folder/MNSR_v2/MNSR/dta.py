import os
from pathlib import Path
from evaluation.ablation import AblationStudy
from evaluation.visualize import ResultsVisualizer

# --- change this to the dataset you were on when it shut down ---
dataset_path = "datasets/gsm8k.json"

dataset_name = Path(dataset_path).stem
output_dir = os.path.join("results", dataset_name)
os.makedirs(output_dir, exist_ok=True)

benchmark_file = os.path.join(output_dir, "benchmark_report.json")
ablation_file = os.path.join(output_dir, "ablation_results.json")

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
        output_dir=output_dir
    )
    visualizer.generate_all()
except Exception as e:
    print(f"⚠️ Visualization step warning: {e}")

print(f"\n✅ Steps 4-5 completed for '{dataset_name}'.")