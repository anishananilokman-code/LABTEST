import operator
from typing import List, Dict, Any, Tuple
import streamlit as st

# ============================ #
# 1) RULE ENGINE CONFIGURATION #
# ============================ #
COMPARISONS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}

DEFAULT_CONDITIONS: List[Dict[str, Any]] = [
    {
        "rule_name": "Windows open → turn off AC",
        "rule_priority": 100,
        "trigger_conditions": [
            ["windows_open", "==", True]
        ],
        "ac_action": {
            "mode": "OFF",
            "fan_speed": "LOW",
            "setpoint": None,
            "reason": "Windows are open"
        }
    },
    {
        "rule_name": "No one home → eco mode",
        "rule_priority": 90,
        "trigger_conditions": [
            ["occupancy", "==", "EMPTY"],
            ["temperature", ">=", 24]
        ],
        "ac_action": {
            "mode": "ECO",
            "fan_speed": "LOW",
            "setpoint": 27,
            "reason": "Home empty, save energy"
        }
    },
    {
        "rule_name": "Too cold → turn off AC",
        "rule_priority": 85,
        "trigger_conditions": [
            ["temperature", "<=", 22]
        ],
        "ac_action": {
            "mode": "OFF",
            "fan_speed": "LOW",
            "setpoint": None,
            "reason": "Already cold"
        }
    },
    {
        "rule_name": "Hot & humid (occupied) → strong cooling",
        "rule_priority": 80,
        "trigger_conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 30],
            ["humidity", ">=", 70]
        ],
        "ac_action": {
            "mode": "COOL",
            "fan_speed": "HIGH",
            "setpoint": 23,
            "reason": "Hot and humid"
        }
    },
    {
        "rule_name": "Night (occupied) → sleep mode",
        "rule_priority": 75,
        "trigger_conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["time_of_day", "==", "NIGHT"],
            ["temperature", ">=", 26]
        ],
        "ac_action": {
            "mode": "SLEEP",
            "fan_speed": "LOW",
            "setpoint": 26,
            "reason": "Night comfort"
        }
    },
    {
        "rule_name": "Hot (occupied) → cool",
        "rule_priority": 70,
        "trigger_conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 28]
        ],
        "ac_action": {
            "mode": "COOL",
            "fan_speed": "MEDIUM",
            "setpoint": 24,
            "reason": "Temperature high"
        }
    },
    {
        "rule_name": "Slightly warm (occupied) → gentle cool",
        "rule_priority": 60,
        "trigger_conditions": [
            ["occupancy", "==", "OCCUPIED"],
            ["temperature", ">=", 26],
            ["temperature", "<", 28]
        ],
        "ac_action": {
            "mode": "COOL",
            "fan_speed": "LOW",
            "setpoint": 25,
            "reason": "Slightly warm"
        }
    },
]

# ============================ #
# 2) RULE ENGINE FUNCTIONS     #
# ============================ #
def validate_condition(facts: Dict[str, Any], condition: List[Any]) -> bool:
    field, op, value = condition
    if field not in facts or op not in COMPARISONS:
        return False
    return COMPARISONS[op](facts[field], value)

def check_rule_match(facts: Dict[str, Any], rule: Dict[str, Any]) -> bool:
    return all(validate_condition(facts, cond) for cond in rule["trigger_conditions"])

def execute_rules(
    facts: Dict[str, Any],
    rules: List[Dict[str, Any]]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], Dict[str, Any] | None]:
    matched_rules = [rule for rule in rules if check_rule_match(facts, rule)]

    if not matched_rules:
        best_action = {
            "mode": "OFF",
            "fan_speed": "LOW",
            "setpoint": None,
            "reason": "No matching rules"
        }
        return best_action, [], None

    matched_rules_sorted = sorted(matched_rules, key=lambda r: r["rule_priority"], reverse=True)
    winning_rule = matched_rules_sorted[0]
    best_action = winning_rule["ac_action"]

    return best_action, matched_rules_sorted, winning_rule

# ============================ #
# 3) STREAMLIT UI              #
# ============================ #
st.set_page_config(page_title="Smart AC System", layout="centered")
st.title("Smart Air Conditioner Rule-Based System")
st.caption("Rule-based expert system using IF–THEN rules with priority conflict resolution.")

with st.sidebar:
    st.header("Home Conditions")

    temperature_input = st.number_input("Temperature (°C)", value=22)
    humidity_input = st.number_input("Humidity (%)", value=46)

    occupancy_input = st.selectbox("Occupancy", ["OCCUPIED", "EMPTY"], index=0)

    time_of_day_input = st.selectbox(
        "Time of Day",
        ["MORNING", "AFTERNOON", "EVENING", "NIGHT"],
        index=3
    )

    windows_open_input = st.checkbox("Windows Open", value=False)
    evaluate_button = st.button("Evaluate AC Action")

facts_data = {
    "temperature": temperature_input,
    "humidity": humidity_input,
    "occupancy": occupancy_input,
    "time_of_day": time_of_day_input,
    "windows_open": windows_open_input
}

st.subheader("Input Facts")
st.json(facts_data)

if evaluate_button:
    action_details, matching_rules, winning_rule = execute_rules(facts_data, DEFAULT_CONDITIONS)

    st.subheader("✅ Recommended AC Action")
    st.success(
        f"""
**Mode:** {action_details['mode']}  
**Fan Speed:** {action_details['fan_speed']}  
**Setpoint:** {action_details['setpoint']}  
**Reason:** {action_details['reason']}
"""
    )

    st.subheader("🏁 Winning Rule (Highest Priority Match)")
    if winning_rule is None:
        st.write("No rule matched. Default action applied.")
    else:
        st.write(f"**{winning_rule['rule_name']}** (Priority: {winning_rule['rule_priority']})")
        st.write("**Trigger Conditions:**")
        for cond in winning_rule["trigger_conditions"]:
            st.write(f"- {cond[0]} {cond[1]} {cond[2]}")

    st.subheader("Matched Rules (Ordered by Priority)")
    if not matching_rules:
        st.info("No rules matched.")
    else:
        for rule in matching_rules:
            st.write(f"• **{rule['rule_name']}** (Priority: {rule['rule_priority']})")
