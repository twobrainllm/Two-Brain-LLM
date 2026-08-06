from pathlib import Path

print("Benchmark Started")

scenario_dir = Path("evaluation/scenarios")

for file in scenario_dir.iterdir():
    print(file.name)