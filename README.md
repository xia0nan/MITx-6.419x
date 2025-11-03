# MITx-6.419x

MITx 6.419x Data Analysis: Statistical Modeling and Computation in Applications

This repository contains coursework and projects for the MITx course 6.419x, focusing on statistical modeling and computational methods in data analysis.

## Repository Structure

- `coding/`: Contains coding assignments and projects.
  - `module3/`: Network analysis project on the CAVIAR dataset, including centrality measures, temporal analysis, HITS algorithm, and structural metrics.
- `notes/`: Course materials and slides.
- `LICENSE`: MIT License.
- `README.md`: This file.

## Module 3: CAVIAR Network Analysis

Module 3 analyzes the CAVIAR wiretap network dataset, consisting of 11 phases of social network data. The project explores network centralities (degree, betweenness, eigenvector), temporal consistency, HITS hubs and authorities, and structural changes in response to external disruptions.

### Data Description
- **Dataset**: CAVIAR wiretap network from 11 two-month phases.
- **Format**: Directed weighted adjacency matrices stored as CSV files (`phase1.csv` to `phase11.csv`).
- **Processing**: Converted to undirected binary graphs for most analyses (ties present if any communication in either direction).
- **Example Data**: Includes Facebook ego-network data (`facebook_combined.txt`) for introductory network analysis exercises.

### Scripts and Tools
- **`main.py`**: Main analysis script for computing centralities, plotting networks, temporal analysis, and HITS algorithm.
- **`project.py`**: Advanced tools for structural metrics, rank series, Kendall tau correlations, and permutation tests.
- **`networks_socialnetwork.py`**: Jupyter notebook (converted to Python) with tutorials on NetworkX basics, centrality measures, and power-law distributions.
- **`answers_f_to_j.md`**: Detailed answers to homework questions (f) through (j), including temporal consistency, event studies, and HITS role analysis.
- **`project_report.md`**: Research report on how police disruptions affected network leadership and structure, with statistical tests and visualizations.

### Key Features
- Compute and compare centralities across phases.
- Temporal analysis of leadership and structure.
- Event study around phase transitions.
- Statistical tests for significance of changes.
- Visualization of network structures and centrality distributions.
- Power-law fitting and goodness-of-fit analysis.

### Setup and Dependencies
- Python 3.10+
- Required packages: `numpy`, `pandas`, `networkx`, `matplotlib`
- Optional: `pygraphviz` for better graph layouts (requires Graphviz installation)

Install dependencies:
```bash
pip install numpy pandas networkx matplotlib
```

For Graphviz support:
```bash
pip install pygraphviz
```

### Usage
Navigate to `coding/module3/` and refer to the detailed README.md there for commands to run analyses, generate plots, and reproduce results.

Example:
```bash
python main.py --summary
```

For the project component:
```bash
python project.py --struct-series --out struct_series.csv
```

### Output Files
The directory contains various output files from analyses, including centrality values, temporal rankings, HITS scores, structural metrics, and plots (e.g., `centralities_phase_4.txt`, `temporal_top_betweenness_ids.txt`, `fig_struct_series.png`).

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
Xiao Nan (Shawn)
