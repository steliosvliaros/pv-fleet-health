# PV Fleet Health (SCADA + Events) – Prototype Package

This repo provides a prototype-ready workflow for:
- SCADA header parsing + canonical signal mapping
- Wide → long normalization
- Time index audit, standard resampling, missing data policy
- Data quality scoring + monitoring confidence
- Irradiance QC + best sensor selection
- Plant-level series building (no double counting)
- Event normalization + taxonomy + overlap merging
- KPI computation + expected-power model + walk-forward validation
- Anomaly detection + curtailment inference + loss accounting
- Fleet benchmarking + clustering + action plan

## Setup

### Conda
```bash
conda env create -f environment.yml
conda activate pv-fleet-health
pip install -e .[ml,pv,dev]
pre-commit install
