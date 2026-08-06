import csv
from pathlib import Path

from two_brain_router.routing import TwoBrainRouter

router = TwoBrainRouter(tier="pc")

scenario_dir = Path("evaluation/scenarios")
results_file = "evaluation/results/results.csv"

with open(results_file, "w", newline="") as f:
    writer = csv.writer(f)

    writer.writerow([
        "scenario",
        "query",
        "route",
        "difficulty",
        "latency_ms",
        "cost_usd",
        "pii_entities"
    ])

    for scenario_file in scenario_dir.glob("*.txt"):
        scenario = scenario_file.stem

        with open(scenario_file) as sf:
            queries = [line.strip() for line in sf if line.strip()]

        for query in queries:
            decision = router.route(query)

            writer.writerow([
                scenario,
                query,
                decision.tier_answered,
                round(decision.difficulty_score, 2),
                round(decision.est_latency_ms, 2),
                round(decision.est_cost_usd, 5),
                decision.pii_entities_masked
            ])

            print(
                f"{scenario}: "
                f"{decision.tier_answered} | "
                f"difficulty={decision.difficulty_score:.2f}"
            )

print(f"\nResults saved to {results_file}")