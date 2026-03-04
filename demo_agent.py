import pandas as pd

from stage_II.features.smart import build_smart_ir, infer_smart_columns
from stage_II.agents.summarizer.summarizer_agent import summarize_ir

DATASET = "dataset/alibaba/test_data/smart.csv"

df = pd.read_csv(DATASET)

smart_cols = infer_smart_columns(df.columns)

print("\n========== KORAL AGENT DEMO ==========\n")

for i in range(len(df)):

    row = df.iloc[i].to_dict()

    print("\n-------------------------------------")
    print("Disk:", row["disk_id"])
    print("Raw telemetry:")
    print(row)

    # Build IR
    ir = {}
    ir.update(build_smart_ir(row, smart_cols))

    print("\nIR SMART FEATURES")
    for x in ir["smart"]:
        print(x["attribute"], "=", x["median"])

    # Run agent
    summary = summarize_ir(ir)

    print("\nAGENT SIGNALS")
    print(summary["signals"])

    # System behavior based on agent
    print("\nSYSTEM INTERPRETATION")

    if "high_wear" in summary["signals"]:
        print("⚠️ High NAND wear detected")

    if "media_errors_present" in summary["signals"]:
        print("⚠️ Flash media errors detected")

    if not summary["signals"]:
        print("✅ No critical reliability signals detected")

print("\n======================================\n")