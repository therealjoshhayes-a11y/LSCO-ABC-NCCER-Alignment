import pandas as pd
from pathlib import Path

traces = [
    't0_core', 't1_electrical', 't2_welding', 't3_pipefitting',
    't4_instrumentation', 't5_carpentry', 't6_rigging', 't7_scaffold'
]

for t in traces:
    path = Path(f'interim/ledger/queues/{t}_queue.csv')
    df = pd.read_csv(path)
    df['module_id'] = df['module_id'].astype(str)
    df.to_csv(path, index=False)
    print(t + ': module_id dtype=' + str(df['module_id'].dtype))
