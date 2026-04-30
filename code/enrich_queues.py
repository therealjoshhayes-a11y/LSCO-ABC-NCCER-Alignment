import pandas as pd
from pathlib import Path

EMBEDDINGS = Path("interim/embeddings")
QUEUES = Path("interim/ledger/queues")

TRACES = [
    "t0_core", "t1_electrical", "t2_welding", "t3_pipefitting",
    "t4_instrumentation", "t5_carpentry", "t6_rigging", "t7_scaffold"
]

# Build WECM lookup: content_key -> outcomes
wecm = pd.read_csv(EMBEDDINGS / "wecm_index.csv", dtype=str)
wecm_lookup = wecm.set_index("content_key")["outcomes"].to_dict()

for trace in TRACES:
    queue_path = QUEUES / f"{trace}_queue.csv"
    index_path = EMBEDDINGS / f"{trace}_index.csv"

    if not queue_path.exists():
        print(f"SKIP {trace} — queue not found")
        continue
    if not index_path.exists():
        print(f"SKIP {trace} — index not found")
        continue

    queue = pd.read_csv(queue_path)
    index = pd.read_csv(index_path, dtype={"module_id": str})

    # Build NCCER lookup: module_id -> objectives
    index["module_id"] = index["module_id"].astype(str)
    nccer_lookup = index.set_index("module_id")["objectives"].to_dict()

    queue["module_id"] = queue["module_id"].astype(str)
    queue["nccer_objectives"] = queue["module_id"].map(nccer_lookup).fillna("")
    queue["wecm_outcomes"] = queue["wecm_course_id"].map(wecm_lookup).fillna("")

    queue.to_csv(queue_path, index=False)
    print(f"OK  {trace} — {len(queue)} rows enriched")

print("\nDone. All queues updated in place.")
