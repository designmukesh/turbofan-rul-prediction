# Data

Raw C-MAPSS files (`*.txt`) are **not committed** — they are reproducible via:

```bash
python data/download_data.py
```

which fetches the PHM Society mirror and extracts it (note: the mirror nests
`CMAPSSData.zip` inside the outer zip; the script handles that).

To reproduce this project's results you only need the **FD001** subset:

| File | Contents |
|---|---|
| `train_FD001.txt` | 100 full run-to-failure engines, 20,631 cycles |
| `test_FD001.txt` | 100 truncated engines (failure point hidden) |
| `RUL_FD001.txt` | True RUL per test engine at truncation |

Full column documentation: [`docs/data_dictionary.md`](../docs/data_dictionary.md).
Dataset citation: Saxena, A. & Goebel, K. (2008), NASA Ames Prognostics
Data Repository.
