# Statistics Y Plot

`statistics-y-plot` is a Python package for statistical analysis and visualization, particularly designed for repeated-measure experimental data.

## Features

- **Repeated-measure t-tests and ANOVA**
  - One-way and two-way repeated-measure ANOVA
  - Post-hoc paired t-tests
  - Automatic removal of NaN values with reporting
  - Significance annotation for p-values (`*`, `**`, `***`)

- **Plotting**
  - Bar or violin plots with scatter and error bars
  - Customizable colors, axis labels, fonts, and tick sizes
  - Option to display y-axis as percentages

- **Data processing utilities**
  - Convert nested list/array data to long-format pandas DataFrame
  - Fully compatible with `statsmodels` and `seaborn`

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/JennyWangw/statistics_Y_plot.git
