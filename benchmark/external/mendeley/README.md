# Mendeley Concrete Crack Images — placeholder

This directory will hold the cross-dataset evaluation against
[Çağlar Özgenel's Mendeley Concrete Crack Images](https://data.mendeley.com/datasets/5y9wdsg2zt)
when the runtime environment can reach `data.mendeley.com` for the
~600 MB download.

For now:

- The `benchmark/external/run_cross_domain.py` script covers the
  per-surface SDNET2018 evaluation, which already probes real
  distribution shift.
- The Mendeley run is the next step (queued behind the network
  capability check).

When this directory becomes active it will contain:

| File | Purpose |
|---|---|
| `download.py` | Fetches the Mendeley zip, verifies checksum, unpacks to `data/external/mendeley/`. |
| `run_eval.py` | Loads each trained checkpoint, evaluates accuracy on Mendeley images (200 cracked + 258 uncracked). |
| `README.md` | How to run; what the expected numbers look like. |
