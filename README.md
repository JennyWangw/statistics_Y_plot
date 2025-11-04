# statistics_Y_plot

`statistics_Y_plot` is a Python package for fast and convenient **statistical analysis and visualization**.  
It is designed for researchers and data scientists who need a clean workflow for t-tests, repeated-measures ANOVA, post-hoc analysis, and high-quality visualization in Python.

---

## Key Features

1. **Comprehensive Statistical Functions**  
   - One-sample t-tests with automatic removal of missing data (NaN)  
   - Independent t-tests with automatic Welch correction when variances are unequal  
   - Paired t-tests with automatic handling of missing pairs  
   - One-way and two-way repeated-measures ANOVA  
   - Automatic post-hoc paired t-tests with Bonferroni correction  

2. **High-quality Visualization**  
   - Bar or violin plots combined with scatter points and error bars  
   - Customizable color palettes and x/y labels  
   - Percent-formatted y-axis for proportions  
   - Automatic significance markers, reference lines, and annotations  

3. **Ease of Use**  
   - Simple, consistent API  
   - Automatically cleans missing data  
   - Automatically selects correct t-test type based on variance equality  
   - Adjustable fonts, tick spacing, and figure sizes for publication-ready plots  

---

## Quick Overview

### 1. One-sample t-test
Automatically removes missing values and reports how many samples were excluded.  
**Example Result:**

| t    | p     | df | mean | std  | N_clean |
|------|-------|----|------|------|---------|
| 2.34 | 0.045 | 3  | 2.20 | 0.95 | 3       |

---

### 2. Independent t-test
Performs standard independent t-test or Welch correction if variances are unequal.  
Reports Levene’s test for variance equality.  
**Example Result:**

| t    | p     | df | equal_var | Levene_stat | Levene_p |
|------|-------|----|-----------|-------------|----------|
| 1.87 | 0.092 | 17 | False     | 2.45        | 0.135    |

---

### 3. Paired t-test
Removes missing pairs automatically and returns t-value, p-value, and degrees of freedom.  
**Example Result:**

| t    | p     | df | n_removed |
|------|-------|----|-----------|
| 2.67 | 0.032 | 3  | 1         |

---

### 4. One-way Repeated-measures ANOVA
Automatically removes subjects with missing values.  
Performs one-way repeated-measures ANOVA, and if significant, conducts post-hoc paired t-tests with Bonferroni correction.  

**Example Output Table:**

| Source    | F Value | Num DF | Den DF | Pr > F | sig |
|-----------|---------|--------|--------|--------|-----|
| Condition | 3.452   | 2      | 9      | 0.073  | n.s. |

*Post-hoc paired t-tests (if significant) will be displayed below.*

---

### 5. Two-way Repeated-measures ANOVA
Supports data with **two nested factors**, automatically removing subjects with missing or incomplete data.  
Performs ANOVA for both main effects and interactions, with post-hoc tests for significant factors.  

**Example Output Table:**

| Source          | F Value | Num DF | Den DF | Pr > F | sig |
|-----------------|---------|--------|--------|--------|-----|
| FactorA         | 5.12    | 2      | 9      | 0.031  | *   |
| FactorB         | 7.89    | 1      | 9      | 0.021  | *   |
| FactorA:FactorB | 1.34    | 2      | 9      | 0.303  | n.s.|

*Post-hoc results are reported for significant main effects.*

---

## Visualization Examples

### Bar Plot with Scatter and Error Bars
- Displays mean ± error, scatter of individual data points  
- Custom color palette and labels  
- Percent-formatted y-axis  
- Reference line and significance annotation  

---

### Advanced Customization
- Adjust figure size, font sizes, tick spacing  
- Rotate x-axis labels  
- Add reference lines, text annotations, or highlight chance levels  
- Automatically handles multiple datasets  

**Example Image Placeholder:**  
![Example Plot](example_plot.png)

---

## Installation

```bash
pip install git+https://github.com/yourusername/statistics_Y_plot.git
