# Public experiment snapshots

This directory contains lightweight JSON snapshots exported from local experiment runs. It includes configurations, epoch metrics, test metrics, fixed generation samples, statistical baselines, and the three-seed aggregate.

Model checkpoints, optimizer states, raw data, and host-specific environment files are intentionally excluded. Recreate checkpoints by following `docs/deployment.md` and the frozen protocol in `configs/experiment_protocol_v1.json`.

Refresh these files after an authorized experiment update:

```powershell
python project.py export-results --run-root runs --output results
```
