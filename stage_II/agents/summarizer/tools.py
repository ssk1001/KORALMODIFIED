#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tools used by the Data Summarizer Agent.

These tools analyze parts of the IR and return structured signals.
"""

from typing import Dict, Any


def analyze_smart_health(ir: Dict[str, Any], row: Dict[str, Any]) -> Dict[str, Any]:

    # If SMART IR exists, use it
    smart = ir.get("smart", [])

    if smart:
        num_attributes = len(smart)
        warning_attrs = 0

        for attr in smart:
            if not isinstance(attr, dict):
                continue

            name = str(attr.get("name", "")).lower()

            if any(x in name for x in ["error", "fail", "uncorrect", "ecc"]):
                warning_attrs += 1

    else:
        # Fallback: inspect dataset columns directly
        warning_attrs = 0
        num_attributes = 0

        for key, value in row.items():

            key_l = key.lower()

            if any(x in key_l for x in ["error", "fail"]):

                num_attributes += 1

                try:
                    if float(value) > 0:
                        warning_attrs += 1
                except:
                    pass

    health = "healthy"

    if warning_attrs >= 3:
        health = "degrading"
    elif warning_attrs >= 1:
        health = "warning"

    return {
        "smart_attribute_count": num_attributes,
        "smart_warning_signals": warning_attrs,
        "device_health_signal": health,
    }