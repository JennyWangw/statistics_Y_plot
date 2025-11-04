from statistics_Y_plot import statistics_Y_plot
import numpy as np

# Initialize the class
stat_plot = statistics_Y_plot(alpha=0.05)

# Example data
data1 = [1.2, 2.3, 3.1, np.nan]
data2 = [1.5, 2.1, 2.9, 3.8]

# one sample t
print(stat_plot.ttest_1samp_with_precheck(data1,popmean=0,nan_policy='omit'))

# ind t
print(stat_plot.ttest_ind_with_precheck(data1,data2))

# paired t 
print(stat_plot.ttest_rel_with_precheck(data1,data2))

# One-way repeated-measure ANOVA
anova_table, posthoc_df = stat_plot.one_factor_rm_anova([data1, data2], factor_name='Condition')

# Plot
stat_plot.plot_with_scatter([data1, data2], xticklabels=['Cond1','Cond2'])


#----------------------- visualization-------------------------------------------------------
np.random.seed(42) 

# Generate three sets of random data
data1 = np.random.normal(loc=0.5, scale=0.2, size=10)  # Mean = 0.5, std = 0.2, 10 samples
data2 = np.random.normal(loc=0.6, scale=0.1, size=10)  # Mean = 0.6, std = 0.1, 10 samples
data3 = np.random.normal(loc=0.4, scale=0.1, size=10)  # Mean = 0.4, std = 0.1, 10 samples

self=stat_plot.plot_with_scatter(data=[data1,data2,data3],
                                plot='bar',
                                palette={'A': 'orange','B':'pink','C':'red'},
                                xlabel='Condition',
                                ylabel='Values',
                                xticklabels=['A','B','C'],
                                figsize=(6,5),fontsize_xlabel=30, fontsize_ylabel=30, fontsize_xtick=28,y_major_locator=0.2,percent_mode=True
)

self.ax.axhline(y=0.5, color='black', linestyle='--', linewidth=5)
self.ax.text(3.6, 0.5, 'Chance Level: 50%', color='black', fontsize=24, ha='center', va='center', fontdict={'family': 'Arial'})
self.ax.set_ylim(0.3,1)

# ------- significant mark --------
max_y=0.9
p_value=0.026174869948176985
self.ax.plot([0.3, 0.7], [max_y, max_y ], color='black',linewidth=3)
sig_label = 'n.s.' if p_value >= 0.05 else ('**' if 0.01>p_value > 0.001 else ('*' if 0.01<p_value<0.05 else '***') )
self.ax.text(0.5, max_y+0.02 , sig_label, ha='center',fontsize=30, fontdict={'family': 'Arial'})

