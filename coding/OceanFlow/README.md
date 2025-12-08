# OceanFlow: Philippine Archipelago Ocean Current Analysis

## 1. Introduction

The Philippine Archipelago is a fascinating multiscale ocean region. Its geometry is complex, with multiple straits, islands, step shelf-breaks, and coastal features, leading to partially interconnected seas and basins. In this project, we will be studying, understanding, and navigating through the ocean current flows.

### Figure: Snapshot of the Ocean Flow Speed in the Philippine Archipelago

The visualization below shows ocean flow speed patterns in the Philippine Archipelago region, with data ranging from 0 to 50 km/h.

## 2. Dataset Description

You can download the data for this homework here: `gp_homework_data.tar.gz`

The data set consists of ocean flow vectors for time *T* from 1 to 100. The flow in the data is an averaged flow from the surface to either near the bottom at 400m depth, whichever is shallower. It is thus a 2D vector field. The files are organized as follows:

- **Files `*u.csv`**: Contain the horizontal components of the vectors (x-axis direction)
- **Files `*v.csv`**: Contain the vertical components of the vectors (y-axis direction)

The numbers in the file names indicate the time index. For example, files `24u.csv` and `24v.csv` contain the information of the flow at time index 23 for **zero-indexed** arrays (Python) or time index 24 for **one-indexed** array (Matlab).

The file `mask.csv`, if needed, contains a matrix identifying land and water.

## 3. Additional Information about Units

The data were collected in **January 2009**. Flows are given in **kilometers per hour (km/h)** units.

The time interval between the data snapshots is **3 hours**. The first time index (0 for zero-indexed, 1 for one-indexed) will correspond to these problems to the time coordinate of 0 hrs. Thus, for example, `1u.csv` gives data at a time coordinate of 0 hours.

## 4. Coordinate System and Grid Spacing

The grid spacing used is **3 km**. The matrix index (0, 0) will correspond to these problems to the coordinate (0 km, 0km), or the **bottom, left** of the plot. For simplicity, we will not be using longitudes and latitudes in this problem.

### Data Organization:
- **Columns** of the `.csv` files correspond to the **horizontal direction (x-axis)**
- **Rows** of the `.csv` files correspond to the **vertical direction (y-axis)**

## 5. Land Mask Information

Although a mask is provided, you will not need it for these problems. Generally, land is already zeroed out in this data. There are some grid points on the coastlines that are non-zero but would be covered by the mask. For this reason, incorporating the mask into your analysis may produce different results. Understanding and resolving this discrepancy is outside the scope of these problems.

## 6. Data Source

The data has been provided by the MSEAS research group at MIT (http://mseas.mit.edu/). The flow field is from a data-assimilative multiresolution simulation obtained using their MSEAS primitive-equation ocean modeling system. It simulates tidal flows to larger-scale dynamics in the region, assimilating a varied set of gappy observations.

## File Structure

```
OceanFlow/
├── data/
│   ├── 1u.csv through 100u.csv  (horizontal components)
│   ├── 1v.csv through 100v.csv  (vertical components)
│   └── mask.csv                  (land/water mask)
└── README.md                      (this file)
```

## Getting Started

1. Load the data files using your preferred data processing library (e.g., `pandas`, `numpy`)
2. Parse the `.csv` files to extract the vector field components
3. Visualize the flow patterns at different time indices
4. Perform analysis on the temporal and spatial variations of ocean currents

## Notes

- Data units: km/h
- Grid spacing: 3 km
- Time interval: 3 hours
- Time range: 0 to 297 hours (100 time steps)
- Coordinate origin: Bottom-left corner (0 km, 0 km)


## Project Status & Deliverables

All tasks for the OceanFlow analysis have been completed. The solution report is available in `answers.md`.

### Completed Tasks and Artifacts

| Problem | Description | Script | Outputs/Plots |
| :--- | :--- | :--- | :--- |
| **1.a** | Min Variance Point | `solve_1a.py` | (Coordinates in report) |
| **1.b** | Max Velocity Point | `solve_1b.py` | (Coordinates/Time in report) |
| **1.c** | Average Flow Vector | `solve_1c.py` | (Vector in report) |
| **2** | Long-range Correlations | `solve_2.py` | `correlation_map.png` |
| **3.a** | Particle Tracking | `solve_3a.py` | `particles_t{0,100,200,300}.png` |
| **3.b** | Debris Search (Variance) | `solve_3b.py` | `debris_t{48,72,120}.png` |
| **4.a** | GP Parameter Estimation | `solve_4a.py` | `gp_{u,v}_optimization.png` |
| **4.b** | Multi-point GP Analysis | `solve_4b.py` | (Table in report) |
| **4.c** | Tau Sensitivity | `solve_4c.py` | (Table in report) |
| **4.d** | Library Comparison | `solve_4d.py` | (Comparison in report) |
| **5** | GP Interpolation | `solve_5.py` | `gp_interpolation.png` |
| **6.a** | 300-Day Simulation | `solve_6a.py` | `debris_300d_sigma{30,90}.png` |
| **6.b** | Monitoring Stations | `solve_6b.py` | `stations_proposal.png` |

### How to Run
Activate the virtual environment and run any script:
```bash
source ../../.venv/bin/activate
python3 solve_6b.py
```
