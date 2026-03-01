# statistics_Y_plot

`statistics_Y_plot` is a Python package for **fast statistical analysis and publication-quality visualization**.  
It provides easy-to-use wrappers for t-tests, repeated-measures ANOVA, and customized plots.


---

## Installation

```bash
pip install git+https://github.com/JennyWangw/statistics_Y_plot.git

```

## Update
```bash
pip install --upgrade --no-cache-dir git+https://github.com/JennyWangw/statistics_Y_plot.git

```
---

## Key Features

1. **Automatic Statistical Analysis**  
   - Automatically handles missing data  
   - Automatically selects the appropriate t-test or Welch test based on variance equality  
   - Automatically performs post-hoc paired t-tests for ANOVA with Bonferroni correction  
   - Automatically annotates significance in summaries  


2. **High-quality Visualization**  
   - Generates publication-ready plots with scatter points and error bars
   - Designed with refined aesthetics suitable for academic papers
   - Supports percentage-formatted y-axis for proportions
   - Advanced customization: adjustable tick spacing, rotated labels, and flexible axis formatting

![Example Plot](figure1.png)

---
## 🧩 Quick Start

```python
from statistics_Y_plot import statistics_Y_plot
import numpy as np

# Initialize
sp = statistics_Y_plot(alpha=0.05)
```

## One-sample t-test

**Function**

```python
ttest_1samp_with_precheck(data, popmean=0)
```

**Example**

```python
data = [1.2, 2.4, np.nan, 3.1, 2.8]
res = sp.ttest_1samp_with_precheck(data, popmean=2)
print(res)
```

**Output**

```
Removed 1 NaN value(s) out of 5 total samples.
{'t': 1.414, 'p': 0.23, 'df': 3, 'mean': '2.3750', 'std': '0.7937', 'N_clean': 4}
```

---

## Independent t-test

**Function**

```python
ttest_ind_with_precheck(x, y)
```

**Example**

```python
x = [4.1, 5.2, 5.4, np.nan, 6.1]
y = [3.9, 4.2, 5.0, 4.8, 4.5]
res = sp.ttest_ind_with_precheck(x, y)
print(res)
```

**Output**

```
Removed 1 NaN(s) from x, 0 NaN(s) from y
{'t': 1.82, 'p': 0.11, 'df_t': 7.0, 'equal_var': True,
 'levene_stat': 0.01, 'levene_p': 0.91, 'levene_df1': 1, 'levene_df2': 7}
```

---

## Paired t-test

**Function**

```python
ttest_rel_with_precheck(x, y)
```

**Example**

```python
x = [1.2, 1.8, 2.0, np.nan, 2.1]
y = [1.5, 2.2, 2.1, 1.9, 2.5]
res = sp.ttest_rel_with_precheck(x, y)
print(res)
```

**Output**

```
Removed 1 sample(s) due to NaN in x or y
{'t': -3.67, 'p': 0.037, 'df': 3, 'n_removed': 1}
```

---

## One-factor Repeated-measures ANOVA

**Function**

```python
one_factor_rm_anova(data, factor_name='Condition')
```

**Input Format**

Each column = one condition
Each row = one subject

**Example**

```python
data = np.array([
    [1.2, 1.5, 2.0],
    [2.0, 2.3, 2.5],
    [1.8, 2.0, 2.4],
    [np.nan, 1.6, 2.1]
])

aov_table, posthoc = sp.one_factor_rm_anova(data, factor_name='Condition')
```

**Output**

```
deleted 1 subjects (3 subjects left)
           F Value  Num DF  Den DF  Pr > F    sig
Condition    8.75      2.0      4.0   0.037      *
```

---

## Two-factor Repeated-measures ANOVA

**Function**

```python
two_factor_rm_anova(data, factorA_name='FactorA', factorB_name='FactorB')
```

**Input Format**

Nested list → [FactorB][FactorA][Subjects]

**Example**

```python
data = [
    [[1.2, 1.5, 1.7], [1.9, 2.1, 2.0]],  # B1: A1, A2
    [[1.3, 1.4, 1.6], [2.2, 2.3, 2.4]]   # B2: A1, A2
]

aov_table, posthoc = sp.two_factor_rm_anova(data, 'A', 'B')
```

**Output**

```
           F Value  Num DF  Den DF  Pr > F    sig
A             9.03      1.0      2.0   0.033      *
B             7.22      1.0      2.0   0.045      *
A:B           0.80      1.0      2.0   0.46    n.s.
```

---

## Plotting

All plotting functions return `self`, and you can access the matplotlib handles via `self.fig` and `self.ax`.

### 1) Bar/violin + scatter + mean±SEM (`plot_with_scatter`)

```python
import numpy as np

np.random.seed(42)
a = np.random.normal(0.65, 0.2, 50)
b = np.random.normal(0.80, 0.5, 50)
c = np.random.normal(0.40, 0.1, 50)

p_ab = sp.ttest_ind_with_precheck(a, b)["p"]
plt_obj = sp.plot_with_scatter([a, b, c], plot="bar",
                               palette={"A":"orange","B":"pink","C":"red"},
                               xlabel="Condition", ylabel="Values",
                               xticklabels=["A","B","C"],
                               figsize=(6,5), y_major_locator=0.2, percent_mode=True)

# chance line + manual significance (customize as you like)
plt_obj.ax.axhline(y=0.5, color="black", linestyle="--", linewidth=5)
plt_obj.ax.set_ylim(0.3, 1.0)
```

![plot_with_scatter demo](figure1.png)

---

### 2) Paired scatter + connections + half-KDE (`paired_scatter_kde_plot`)

```python
import numpy as np

np.random.seed(42)
a1 = np.random.normal(0.65, 0.2, 50)
a2 = np.random.normal(0.80, 0.5, 50)

plt_obj = sp.paired_scatter_kde_plot([a1, a2],
                                     palette={"A1":"#4a90e2","A2":"#e24a33"},
                                     xticklabels=["A1","A2"],
                                     ylabel="Values",
                                     figsize=(4,6), y_major_locator=0.6,
                                     xlim=(-0.5, 1.5), percent_mode=False)
```

![paired_scatter_kde_plot demo](figure2.png)

---

### 3) Two-factor plot with hue (`two_factor_with_hue`)

Use `data_to_long` to convert nested inputs to long-format first (recommended for two-factor plotting).

```python
import numpy as np

np.random.seed(42)
data_A1B1 = np.random.normal(0.65, 0.1, 50)
data_A2B1 = np.random.normal(0.90, 0.1, 50)
data_A1B2 = np.random.normal(0.35, 0.1, 50)
data_A2B2 = np.random.normal(0.85, 0.1, 50)

df = sp.data_to_long([[data_A1B1, data_A2B1],
                      [data_A1B2, data_A2B2]],
                     factorA_name="condition A", factorB_name="condition B",
                     factorA_levels=["A1","A2"], factorB_levels=["B1","B2"])

plt_obj = sp.two_factor_with_hue(df, x="condition A", y="Value", hue="condition B",
                                 plot="bar", xticklabels=["A1","A2"],
                                 xlabel="Condition", ylabel="Value",
                                 y_major_locator=0.5)
```

![two_factor_with_hue demo](figure3.png)
---

## Helper Function

**Convert nested data to long format**

```python
nested = [
    [[1,2,3], [2,3,4]],
    [[3,4,5], [5,6,7]]
]
df_long = sp.data_to_long(nested, 'A', 'B')
print(df_long.head())
```

**Output**

```
  Subject  A  B  Value
0      S1 A1 B1      1
1      S2 A1 B1      2
2      S3 A1 B1      3
3      S1 A2 B1      2
4      S2 A2 B1      3
```

---

## Summary

| Function                    | Purpose                                | Input Type     | Output                 |
| --------------------------- | -------------------------------------- | -------------- | ---------------------- |
| `ttest_1samp_with_precheck` | One-sample t-test with NaN check       | array          | dict                   |
| `ttest_ind_with_precheck`   | Independent t-test with Levene’s test  | 2 arrays       | dict                   |
| `ttest_rel_with_precheck`   | Paired t-test                          | 2 arrays       | dict                   |
| `one_factor_rm_anova`       | One-way repeated measures ANOVA        | 2D array       | (DataFrame, DataFrame) |
| `two_factor_rm_anova`       | Two-way repeated measures ANOVA        | nested list    | (DataFrame, dict)      |
| `plot_with_scatter`         | Plot bar/violin + scatter              | list of arrays | matplotlib object      |
| `data_to_long`              | Convert nested lists to long DataFrame | list           | DataFrame              |



