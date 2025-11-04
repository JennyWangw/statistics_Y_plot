from my_package import statistics_Y_plot
import numpy as np

# Initialize the class
stat_plot = statistics_Y_plot(alpha=0.05)

# Example data
data1 = [1.2, 2.3, 3.1, 4.0]
data2 = [1.5, 2.1, 2.9, 3.8]

# One-way repeated-measure ANOVA
anova_table, posthoc_df = stat_plot.one_factor_rm_anova(data1, data2, factor_name='Condition')

# Plot
stat_plot.plot_with_scatter([data1, data2], xticklabels=['Cond1','Cond2'])
