from typing import Dict, Any, List

from stage_II.llm.openai_client import OpenAIChatClient


def extract_smart_signals(ir: Dict[str, Any]) -> List[str]:

    signals = []
    smart = ir.get("smart", [])

    wear = next((x for x in smart if x["attribute"] == "wear_leveling_count"), None)
    media = next((x for x in smart if x["attribute"] == "media_errors"), None)
    read_err = next((x for x in smart if x["attribute"] == "read_errors"), None)
    write_err = next((x for x in smart if x["attribute"] == "write_errors"), None)

    if wear and wear["median"] > 60:
        signals.append("high_nand_wear")

    if media and media["median"] > 0:
        signals.append("media_errors_detected")

    if read_err and read_err["median"] > 10:
        signals.append("high_read_error_rate")

    if write_err and write_err["median"] > 10:
        signals.append("high_write_error_rate")

    return signals


def llm_explain_signals(signals: List[str], telemetry: Dict[str, float]) -> str:

    if not signals:
        return "No reliability risks detected in SMART telemetry."

    system_prompt = """
You are an SSD reliability engineer.

Explain what the detected SMART telemetry signals imply about device health.
Focus on NAND wear, flash reliability, and potential failure risk.
Respond in 1–2 concise sentences.
"""

    user_prompt = f"""
Detected SMART signals:
{signals}

SMART telemetry values:
{telemetry}

Explain the likely SSD health condition.
"""

    client = OpenAIChatClient()

    response = client.chat(
        system=system_prompt,
        user=user_prompt,
        temperature=0.2,
        max_tokens=120
    )

    return response.text.strip()


def summarize_ir(ir: Dict[str, Any]) -> Dict[str, Any]:

    summary = {}

    smart = ir.get("smart", [])

    signals = extract_smart_signals(ir)

    telemetry = {x["attribute"]: x["median"] for x in smart}

    summary["signals"] = signals

    if signals:
        summary["ai_explanation"] = llm_explain_signals(signals, telemetry)
    else:
        summary["ai_explanation"] = "SMART telemetry does not indicate reliability issues."

    return summary