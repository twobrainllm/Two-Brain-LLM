import csv

rows = []

with open("evaluation/results/results.csv") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

total = len(rows)
local_routes = sum(1 for r in rows if r["route"] == "local")
cloud_routes = sum(1 for r in rows if r["route"] == "cloud")

avg_latency = sum(float(r["latency_ms"]) for r in rows) / total
avg_cost = sum(float(r["cost_usd"]) for r in rows) / total
total_pii = sum(int(r["pii_entities"]) for r in rows)

print("===== SUMMARY =====")
print(f"Total Queries: {total}")
print(f"Local Routes: {local_routes}")
print(f"Cloud Routes: {cloud_routes}")
print(f"Average Latency: {avg_latency:.2f} ms")
print(f"Average Cost: ${avg_cost:.5f}")
print(f"PII Entities Masked: {total_pii}")