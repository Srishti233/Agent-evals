"""Regenerates data/eval_set.json (kept in the repo so the dataset is reviewable/editable)."""
import json
from pathlib import Path

D = {k: k for k in [
    "01_company_overview", "02_product_atlas", "03_product_sparrow", "04_product_beacon", "05_leave_policy",
    "06_parental_leave", "07_remote_work", "08_expenses", "09_learning_budget", "10_onboarding",
    "11_engineering_practices", "12_security_policy", "13_manufacturing", "14_sales_pricing", "15_support",
    "16_safety_standards", "17_roadmap", "18_customers"]}

CASES = [
 # ---- factual (16) ----
 ("factual", "Who founded Lumora Labs?", "Anika Rao and Dev Malhotra.", ["01_company_overview"]),
 ("factual", "In which year was Lumora Labs founded?", "2016.", ["01_company_overview"]),
 ("factual", "What is the payload capacity of the Atlas robot?", "1,200 kg.", ["02_product_atlas"]),
 ("factual", "How long does a full charge of the Atlas battery take?", "90 minutes.", ["02_product_atlas"]),
 ("factual", "How many items per hour can the Sparrow robot pick?", "Up to 600 items per hour.", ["03_product_sparrow"]),
 ("factual", "How much does the Beacon subscription cost?", "$150 per robot per month.", ["04_product_beacon"]),
 ("factual", "How many days of paid annual leave do full-time employees get?", "24 days per year.", ["05_leave_policy"]),
 ("factual", "How many weeks of paid parental leave do primary caregivers receive?", "26 weeks.", ["06_parental_leave"]),
 ("factual", "On which days are employees expected to be in the office?", "Tuesday, Wednesday and Thursday.", ["07_remote_work"]),
 ("factual", "Within how many days must expense claims be submitted?", "Within 30 days.", ["08_expenses"]),
 ("factual", "How large is the yearly learning budget per employee?", "$1,200 per year.", ["09_learning_budget"]),
 ("factual", "How long is the probation period for new hires?", "90 days (3 months).", ["10_onboarding"]),
 ("factual", "How many approvals does a pull request need before it can be merged?", "Two approvals.", ["11_engineering_practices"]),
 ("factual", "What is the minimum password length?", "At least 14 characters.", ["12_security_policy"]),
 ("factual", "Which safety standard is the Atlas certified to?", "ISO 3691-4.", ["16_safety_standards"]),
 ("factual", "How long is the standard warranty?", "24 months.", ["14_sales_pricing"]),
 # ---- multi-hop (8) ----
 ("multi-hop", "What is the payload of the robot that is assembled at the Chennai plant?", "5 kg (the Sparrow).", ["13_manufacturing", "03_product_sparrow"]),
 ("multi-hop", "How long is the burn-in test for the type of robot that Kestrel Logistics runs?", "72 hours (Kestrel runs Atlas robots).", ["18_customers", "13_manufacturing"]),
 ("multi-hop", "What is the list price of the robot that Bharat Mart operates?", "$9,500 per unit (the Sparrow).", ["18_customers", "03_product_sparrow"]),
 ("multi-hop", "By how much will the payload of Atlas 2 exceed the payload of the current Atlas?", "600 kg (1,800 kg versus 1,200 kg).", ["17_roadmap", "02_product_atlas"]),
 ("multi-hop", "In which programming language is the firmware written for the robot that moves pallets?", "Rust (Atlas moves pallets; robot firmware is written in Rust).", ["02_product_atlas", "11_engineering_practices"]),
 ("multi-hop", "What is the top speed of the robot model that Kestrel Logistics operates?", "2.0 m/s (Kestrel runs Atlas robots).", ["18_customers", "02_product_atlas"]),
 ("multi-hop", "In which city did Lumora Labs open an office in 2021, and how many employees is the planned Munich office targeting?", "Austin, Texas; the Munich office targets 60 employees.", ["01_company_overview", "17_roadmap"]),
 ("multi-hop", "Which assembly site has the higher monthly capacity, and which robot does it build?", "The Chennai plant, which builds Sparrow robots (800 units per month versus 300 in Pune).", ["13_manufacturing"]),
 # ---- math (8) ----
 ("math", "What would 10 Atlas robots cost at list price, before any discount?", "$240,000.", ["02_product_atlas"]),
 ("math", "What is the price of 25 Atlas robots after the applicable volume discount?", "$540,000 (25 x $24,000 = $600,000, minus 10%).", ["02_product_atlas", "14_sales_pricing"]),
 ("math", "How much does Beacon cost per month for a site that runs 120 robots?", "$18,000 per month.", ["04_product_beacon"]),
 ("math", "How many items can one Sparrow pick during an 8-hour shift at its maximum rate?", "4,800 items.", ["03_product_sparrow"]),
 ("math", "An employee has already used 15 days of annual leave this year. How many annual leave days remain?", "9 days.", ["05_leave_policy"]),
 ("math", "How many Atlas units can the Pune factory produce in a year at full capacity?", "3,600 units.", ["13_manufacturing"]),
 ("math", "What is the total price of 6 Sparrow robots after the applicable volume discount?", "$54,150 (6 x $9,500 = $57,000, minus 5%).", ["03_product_sparrow", "14_sales_pricing"]),
 ("math", "How much on-call allowance does an engineer earn for 4 on-call weeks?", "$1,400.", ["11_engineering_practices"]),
 # ---- unanswerable (8) ----
 ("unanswerable", "What was Lumora Labs' total revenue in 2025?", "I don't know.", []),
 ("unanswerable", "Who is the Chief Financial Officer of Lumora Labs?", "I don't know.", []),
 ("unanswerable", "What is the top speed of the Sparrow robot?", "I don't know.", []),
 ("unanswerable", "How many days of bereavement leave do employees get?", "I don't know.", []),
 ("unanswerable", "What will the list price of the Atlas 2 be?", "I don't know.", []),
 ("unanswerable", "Which cloud provider hosts the Beacon software?", "I don't know.", []),
 ("unanswerable", "What is Lumora Labs' stock ticker symbol?", "I don't know.", []),
 ("unanswerable", "How many employees work in the Austin office?", "I don't know.", []),
]

counters: dict[str, int] = {}
out = []
for cat, q, a, docs in CASES:
    counters[cat] = counters.get(cat, 0) + 1
    out.append({"id": f"{cat[:4]}-{counters[cat]:02d}", "category": cat, "question": q,
                "gold_answer": a, "supporting_doc_ids": docs})
for c in out:
    for d in c["supporting_doc_ids"]:
        assert d in D, d
Path(__file__).resolve().parent.parent.joinpath("data/eval_set.json").write_text(
    json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(len(out), {k: v for k, v in counters.items()})
