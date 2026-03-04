import json

path = "stage_II/runs/demo_run/responses.jsonl"

with open(path) as f:
    for line in f:
        data = json.loads(line)

        if data["task"] == "descriptive" and data["sample_id"] == "s1":
            print(json.dumps(data, indent=2))