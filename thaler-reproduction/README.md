# Econ 148 Track B Reproduction Package

Paper: Michael Thaler (2024), "The Fake News Effect: Experimentally Identifying Motivated Reasoning Using Trust in News," *American Economic Journal: Microeconomics*.

## How to run

### Option 1: Colab/Jupyter
Open `notebooks/thaler_reproduction_notebook.ipynb` and run all cells. The notebook expects the repository folders to remain in the same structure:

- `data/`: public replication CSV files
- `src/`: Python analysis script
- `figures/`: generated figures
- `tables/`: generated table outputs
- `report/`: LaTeX and PDF report
- `original_stata/`: original author-provided Stata replication scripts and README

### Option 2: Python script
From the root folder, run:

```bash
python src/analysis.py
```

## Outputs

The script/notebook writes reproduced figures to `figures/` and reproduced regression/summary tables to `tables/`.

## Notes

This Python reproduction focuses on the core Track B requirements: data documentation, summary statistics, original equations, main figures, main regression table, and discussion of replication hurdles. The original Stata scripts are included for full author-package transparency.
