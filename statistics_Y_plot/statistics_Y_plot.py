import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FuncFormatter
from statsmodels.stats.anova import AnovaRM
from itertools import combinations

import warnings
warnings.filterwarnings('ignore')

class statistics_Y_plot:
    def __init__(self, alpha=0.05):
        """
        Initialize the class with a significance threshold.

        Parameters
        ----------
        alpha : float, optional
            Significance level for statistical tests (default is 0.05).
        """

        self.alpha = alpha

    # -------------------- Utility Functions --------------------
    @staticmethod
    def p_to_sig(p):
        """
        Convert a p-value to a significance annotation.

        Parameters
        ----------
        p : float
            The p-value to convert.

        Returns
        -------
        str
            Significance string: '***' for p<0.001, '**' for p<0.01, '*' for p<0.05, 
            'n.s.' for not significant.
        """
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'
    
    @staticmethod
    def data_to_long(data, factorA_name='FactorA', factorB_name='FactorB', subject_prefix='S',factorA_levels=None,factorB_levels=None):
        """
        Convert nested lists or arrays into a long-format pandas DataFrame.
        
        Assumes a TWO-level nesting corresponding to two factors:
        - Outer list/array -> levels of Factor B:c or d
        - Inner list/array -> levels of Factor A:c1,c2,c3 or d1,d2,d3
        e.g. data=[[c1,c2,c3],[d1,d2,d3]]
        
        Parameters
        ----------
        data : list of lists or arrays
            Nested data structure with shape [factorB][factorA][subjects].
        factorA_name : str, optional
            Column name for the first factor (default 'FactorA').
        factorB_name : str, optional
            Column name for the second factor (default 'FactorB').
        subject_prefix : str, optional
            Prefix for subject labels (default 'S').
        factorA_levels: list, optinal
            default:['A1','A2',...]
        factorB_levels: list, optinal
            default:['B1','B2',...]
        Returns
        -------
        pandas.DataFrame
            Long-format DataFrame with columns: 'Subject', factorA_name, factorB_name, 'Value'.
        """

        long_data = []
        nB = len(data)
        nA = len(data[0]) if nB > 0 else 0
        
        # default A1/A2..., B1/B2...
        if factorA_levels is None:
            factorA_levels = [f'A{i+1}' for i in range(nA)]
        if factorB_levels is None:
            factorB_levels = [f'B{i+1}' for i in range(nB)]
    
        for b_idx, b_level in enumerate(data, start=0):
            for a_idx, a_list in enumerate(b_level, start=0):
                if not hasattr(a_list, '__iter__') or isinstance(a_list, (str, bytes)):
                    a_list = [a_list]
                for sub_idx, value in enumerate(a_list, start=1):
                    long_data.append({
                        'Subject': f'{subject_prefix}{sub_idx}',
                        factorA_name: factorA_levels[a_idx],
                        factorB_name: factorB_levels[b_idx],
                        'Value': value
                    })
        return pd.DataFrame(long_data)

    def run_pairedT_for_posthoc(self, df_long, factor, ttest_func):
        """
        - Perform post-hoc paired t-tests for all levels of a given factor.
        - For single factor with >2 levels: all pairwise comparisons with Bonferroni.
        - For interaction: identify factor with 2 levels (split_factor). 
            Then for each level of the other factor (test_factor), perform paired t-tests
            between the two levels of split_factor for corresponding subjects.
    
        Parameters
        ----------
        df_long : pandas.DataFrame
            Long-format DataFrame containing the data.
        factor : str
            Column name of the factor to test. This factor should have multiple levels.
        ttest_func : callable
            Function to perform the paired t-test. Should accept two arrays and return a p-value.
    
        Returns
        -------
	pandas.DataFrame
            DataFrame with columns:
            - 'Level1', 'Level2': compared factor levels
            - 't': t-statistic
            - 'p-unc': uncorrected p-value
            - 'p-bonf': Bonferroni-corrected p-value
            - 'sig': significance annotation ('***', '**', '*', 'n.s.')

        """

        if ":" not in factor:
            levels = df_long[factor].unique() # Get all levels of this factor
            results = []
            for a, b in combinations(levels, 2):
                vals_a = df_long[df_long[factor] == a].sort_values('Subject')['Value'].values
                vals_b = df_long[df_long[factor] == b].sort_values('Subject')['Value'].values
                # filter NaN
                mask = ~np.isnan(vals_a) & ~np.isnan(vals_b)
                vals_a, vals_b = vals_a[mask], vals_b[mask]
                res_t = ttest_func(vals_a, vals_b)
                p_corr = min(res_t['p'] * len(list(combinations(levels, 2))), 1.0)
                results.append({
                    'Level1': a,
                    'Level2': b,
                    't': res_t['t'],
                    'p-unc': res_t['p'],
                    'p-bonf': p_corr,
                    'sig': self.p_to_sig(p_corr)
                })
            return pd.DataFrame(results)
        else:#interaction
            f1, f2 = factor.split(':')
            levels_f1 = df_long[f1].unique()
            levels_f2 = df_long[f2].unique()
            if len(levels_f1) == 2:
                split_factor, test_factor = f1, f2
            elif len(levels_f2) == 2:
                split_factor, test_factor = f2, f1
            else:
                return None  # skip if neither factor has 2 levels
    
            results = []
            for test_level in df_long[test_factor].unique():
                df_sub = df_long[df_long[test_factor] == test_level]
                split_levels = df_sub[split_factor].unique()
                if len(split_levels) != 2:
                    continue  # skip if not 2 levels in this subset
            
                # 对齐 Subject
                df_a = df_sub[df_sub[split_factor] == split_levels[0]].set_index('Subject')
                df_b = df_sub[df_sub[split_factor] == split_levels[1]].set_index('Subject')
                common_subjects = df_a.index.intersection(df_b.index)
                vals_a = df_a.loc[common_subjects, 'Value'].values
                vals_b = df_b.loc[common_subjects, 'Value'].values
                mask = ~np.isnan(vals_a) & ~np.isnan(vals_b)
                vals_a, vals_b = vals_a[mask], vals_b[mask]
                
                res_t = ttest_func(vals_a, vals_b)
                p_corr = min(res_t['p'] * 1, 1.0)  # one comparison per test_level
                results.append({
                    split_factor+'_level': test_level,
                    'Level1': levels_f1[0] if split_factor == f1 else levels_f2[0],
                    'Level2': levels_f1[1] if split_factor == f1 else levels_f2[1],
                    't': res_t['t'],
                    'p-unc': res_t['p'],
                    'p-bonf': p_corr,
                    'sig': self.p_to_sig(p_corr)
                })
            return pd.DataFrame(results)

    def ttest_1samp_with_precheck(self, data, popmean=0, nan_policy='omit', return_summary=True):
        """
        - Perform a one-sample t-test with automatic NaN removal, and report removed counts
        ----------
        data : array-like
            Sample data (can include NaN).
        popmean : float
            Population mean for comparison.
        nan_policy : {'omit', 'propagate', 'raise'}, optional
            How to handle NaNs.
        return_summary : bool, optional
            If True, also return mean, std, and n.
    
        Returns
        -------
        result : dict
            Contains t-statistic, p-value, df, and optionally summary statistics.
        """
        data = np.array(data)
        data_clean = data[~np.isnan(data)]
        df = len(data_clean) - 1

        if np.isnan(data).sum() > 0:
            print(f" Removed {len(data_clean)} NaN value(s) out of {len(data)} total samples.")
            
        res = stats.ttest_1samp(data_clean, popmean=popmean, nan_policy=nan_policy)
        
        result = {
            't': res.statistic,
            'p': res.pvalue,
            'df': df
        }
        
        if return_summary:
            result.update({
                'mean': f"{np.mean(data_clean):.4f}",
                'std': f"{np.std(data_clean, ddof=1):.4f}", 
                'N_clean': len(data_clean) #number of sample after nan removal 
            })
        
        return result


            

    def ttest_ind_with_precheck(self, x, y):
        """
        Perform independent t-test with automatic NaN removal and variance check.

        Steps:
        1. Remove NaNs and report removed counts.
        2. Check variance homogeneity using Levene's test.
           - If unequal, use Welch's t-test.
           - If equal, use standard independent t-test.
        3. Return t-statistic, p-value, df, and Levene's test results.

        Parameters
        ----------
        x, y : array-like
            Two independent samples.

        Returns
        -------
        dict
            Keys:
            - 't', 'p': t-test statistic and p-value
            - 'df_t': degrees of freedom for the t-test/welch
            - 'equal_var': True if variances are equal
            - 'levene_stat', 'levene_p': Levene test statistic and p-value
            - 'levene_df1', 'levene_df2': Levene degrees of freedom
        """

        x, y = np.array(x), np.array(y)
        nan_removed_x = np.isnan(x).sum()
        nan_removed_y = np.isnan(y).sum()
        x, y = x[~np.isnan(x)], y[~np.isnan(y)] #nan removal
        if nan_removed_x > 0 or nan_removed_y > 0:# report removed ccounts（only when nan exits）
            print(f"Removed {nan_removed_x} NaN(s) from x, {nan_removed_y} NaN(s) from y")

        lev_stat, lev_p = stats.levene(x, y)
        n1, n2 = len(x), len(y)
        df1_lev = 2 - 1       # k-1
        df2_lev = n1 + n2 - 2 # N-k
        equal_var = lev_p > self.alpha
        res = stats.ttest_ind(x, y, equal_var=equal_var) #ind t test /Welch
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        if equal_var:
            df_t = n1 + n2 - 2
        else:
            df_t = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        return {
            't': res.statistic,'p': res.pvalue,'df_t': df_t,#自由度（适用t/Welch）
            'equal_var': equal_var,#True:方差齐性
            'levene_stat': lev_stat, 'levene_p': lev_p,'levene_df1': df1_lev,'levene_df2': df2_lev
        }


    def ttest_rel_with_precheck(self, x, y):
        """
        Perform a paired t-test with automatic removal of NaN-containing pairs, and report removed counts.

        Parameters
        ----------
        x, y : array-like
            Paired samples.

        Returns
        -------
        dict
            Contains t-statistic, p-value, degrees of freedom, and number of removed pairs due to NaNs.
        """
        x, y = np.array(x), np.array(y)
        mask = ~np.isnan(x) & ~np.isnan(y)
        n_removed = len(x) - np.sum(mask)
        x_clean, y_clean = x[mask], y[mask]# remove subject with NaN values
        if n_removed > 0:
            print(f"Removed {n_removed} sample(s) due to NaN in x or y")
        res = stats.ttest_rel(x_clean, y_clean)
        df = len(x_clean) - 1# df= sample N - 1
        return {
            't': res.statistic,
            'p': res.pvalue,
            'df': df,
            'n_removed': n_removed
        }

    
    
    def one_factor_rm_anova(self, data, factor_name='Condition', subject_prefix='S'):
        """
        Perform one-way repeated-measures ANOVA with optional post-hoc paired t-tests.

        Steps:
        1. Remove subjects with any missing values and report the number removed.
        2. Perform one-factor repeated-measures ANOVA.
        3. If the ANOVA is significant, run post-hoc paired t-tests with Bonferroni correction.

        Parameters
        ----------
        *conditions : array-like
            Each argument represents one condition (list or array of subject data).
        factor_name : str, optional
            Name of the factor (default 'Condition').
        subject_prefix : str, optional
            Prefix for subject IDs (default 'S').

        Returns
        -------
        tuple
            (anova_table, posthoc_df) where posthoc_df is None if no post-hoc tests were performed.
        """
        data = np.array(data)
        n_before = data.shape[0]
        
        mask = ~np.isnan(data).any(axis=1)# remove subject with NaN values
        data_clean = data[mask]
        n_after = data_clean.shape[0]
        n_dropped = n_before - n_after
        if n_dropped!=0:
            print(f'deleted {n_dropped} subjects ({n_after} subjects left)')

        n_cond = data_clean.shape[1]
        subjects = [f'{subject_prefix}{i+1}' for i in range(n_after)]
        df_long = pd.DataFrame({
            'Subject': np.repeat(subjects, n_cond),
            factor_name: np.tile([f'Level{i+1}' for i in range(n_cond)], n_after),
            'Value': data_clean.flatten()
        })

        #-------anova-------
        aov = AnovaRM(df_long, depvar='Value', subject='Subject', within=[factor_name]).fit() #anova
        aov_table = aov.anova_table.copy()
        aov_table['sig'] = aov_table['Pr > F'].apply(self.p_to_sig)
        print(aov.anova_table)

        #------post hoc------
        posthoc_df = None
        if aov_table['Pr > F'][0] < self.alpha:
            print('\nPost-hoc')
            posthoc_df = self.run_pairedT_for_posthoc(df_long, factor_name, self.ttest_rel_with_precheck)
            print(posthoc_df)
            
        return aov_table, posthoc_df


    
    def two_factor_rm_anova(self, data, factorA_name='FactorA', factorB_name='FactorB', subject_prefix='S'):
        """
        Perform two-way repeated-measures ANOVA for nested data (two factors, possibly multi-level).

        Steps:
        1. Convert nested data to long-format DataFrame.
        2. Remove subjects with any missing or incomplete data and report how many were removed.
        3. Perform two-way repeated-measures ANOVA.
        4. If main effects are significant, perform post-hoc paired t-tests for simple effects.

        Parameters
        ----------
        data : list of lists or arrays
            Nested data structure: outer list corresponds to factor B levels, inner lists correspond to factor A levels.
        factorA_name : str, optional
            Name of factor A (default 'FactorA').
        factorB_name : str, optional
            Name of factor B (default 'FactorB').
        subject_prefix : str, optional
            Prefix for subject IDs (default 'S').

        Returns
        -------
        tuple
            (anova_table, posthoc_dict) where posthoc_dict contains post-hoc DataFrames for significant factors or None.
        """
        df_long = self.data_to_long(data, factorA_name, factorB_name, subject_prefix) 

        subjects_total = df_long['Subject'].unique()
        factorA_levels = df_long[factorA_name].unique()
        factorB_levels = df_long[factorB_name].unique()
        valid_subjects = []
        for sub in subjects_total:
            df_sub = df_long[df_long['Subject'] == sub]
            if not df_sub['Value'].isna().any() and \
               len(df_sub) == len(factorA_levels) * len(factorB_levels):  # condi 1: without NaN； condi 2: valid value for every factorA×factorB combo
                valid_subjects.append(sub)
    
        n_dropped = len(subjects_total) - len(valid_subjects)
        if n_dropped > 0:
            print(f"Deleted {n_dropped} subjects due to missing values or incomplete data ({len(valid_subjects)} subjects left)")
    
        df_long_clean = df_long[df_long['Subject'].isin(valid_subjects)]


        aov = AnovaRM(df_long_clean, depvar='Value', subject='Subject', within=[factorA_name, factorB_name]).fit() #anova
        aov_table = aov.anova_table.copy()
        aov_table['sig'] = aov_table['Pr > F'].apply(self.p_to_sig)
        print(aov_table)

        posthoc_dict = {} #post-hoc
        for factor in [factorA_name, factorB_name,f'{factorA_name}:{factorB_name}']:
            if aov_table.loc[factor, 'Pr > F'] < self.alpha:
                print(f'\nPost-hoc for {factor}')
                posthoc_dict[factor] = self.run_pairedT_for_posthoc(df_long_clean, factor, self.ttest_rel_with_precheck)
                print(posthoc_dict[factor])
            else:
                posthoc_dict[factor] = None

        return aov_table, posthoc_dict



    #———————————————————————visualization—————————————————————————————————————————————
    def set_plot_style(self, ax, xlabel='', ylabel='', xticklabels=None,
                       xlabelpad=15,fontsize_xlabel=30, fontsize_ylabel=30, fontsize_xtick=28,fontsize_ytick=28,
                       spine_width=5, tick_length=15, y_major_locator=3,x_rotation=0,percent_mode=None):

        """
        Standardize the styling of a matplotlib Axes object.
    
        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes object to style.
        xlabel : str, optional
            Label for the x-axis (default '').
        ylabel : str, optional
            Label for the y-axis (default '').
        xticklabels : list of str or None, optional
            Custom labels for x-axis ticks (default None).
        xlabelpad : float, optional
            Padding between x-axis label and axis in points (default 15).
        fontsize_xlabel : int, optional
            Font size of x-axis label (default 30).
        fontsize_ylabel : int, optional
            Font size of y-axis label (default 30).
        fontsize_xtick : int, optional
            Font size of x-axis tick labels (default 28).
        fontsize_ytick : int, optional
            Font size of y-axis tick labels (default 28).
        spine_width : float, optional
            Width of axis spines (default 5).
        tick_length : float, optional
            Length of ticks (default 15).
        y_major_locator : float, optional
            Spacing between major y-axis ticks (default 3).
        x_rotation : float, optional
            Rotation angle of x-axis tick labels in degrees (default 0).
        percent_mode : {'percent', None}, optional
            If 'percent', format y-axis labels as percentages (default None).
    
        Returns
        -------
        None
        """
        ax.set_xlabel(xlabel, fontsize=fontsize_xlabel, fontdict={'family': 'Arial'}, labelpad=xlabelpad)
        ax.set_ylabel(ylabel, fontsize=fontsize_ylabel, fontdict={'family': 'Arial'})
        
        if xticklabels is not None:
            ax.set_xticks(range(len(xticklabels)), labels=xticklabels,rotation=x_rotation)
        ax.tick_params(axis='x', labelsize=fontsize_xtick)
        ax.tick_params(axis='y', labelsize=fontsize_ytick)

        # Axis spines
        ax.spines['left'].set_linewidth(spine_width)
        ax.spines['bottom'].set_linewidth(spine_width)

        # Axis ticks
        ax.tick_params(axis='both', which='major', width=spine_width, length=tick_length)

        # Set y-axis major tick spacing
        ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))

        # Format y-axis as percentage if requested
        def percent_formatter(x, pos):
            return f'{x*100:.0f}%' 
        if percent_mode:
            ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))


        sns.despine()
        plt.tight_layout()


    def plot_with_scatter(
        self, data, palette=None, p_value=None,plot='bar',
        xlabel='', ylabel='', xticklabels=None, figsize=(6,5),
        fontsize_xlabel=30, fontsize_ylabel=30, fontsize_xtick=28,fontsize_ytick=28,y_major_locator=3,x_rotation=0,percent_mode=None,):
        """
        Create a combined plot with bar/violin, scatter, and error bars.
    
        Parameters
        ----------
        data : array-like or DataFrame
            Data to plot.
        palette : dict, optional
            Mapping from x-axis categories to colors, e.g., {'A': 'red', 'B': 'green', 'C': 'blue'}.
        p_value : float, optional
            p-value for significance annotation on the plot.
        plot : {'bar', 'violin'}, optional
            Type of main plot (default 'bar').
        xlabel : str, optional
            Label for the x-axis (default '').
        ylabel : str, optional
            Label for the y-axis (default '').
        xticklabels : list of str, optional
            Custom labels for x-axis ticks (default None).
        figsize : tuple, optional
            Figure size in inches (width, height) (default (6, 5)).
        fontsize_xlabel : int, optional
            Font size for x-axis label (default 30).
        fontsize_ylabel : int, optional
            Font size for y-axis label (default 30).
        fontsize_xtick : int, optional
            Font size for x-axis tick labels (default 28).
        fontsize_ytick : int, optional
            Font size for y-axis tick labels (default 28).
        y_major_locator : float, optional
            Spacing between major y-axis ticks (default 3).
        x_rotation : float, optional
            Rotation angle of x-axis tick labels in degrees (default 0).
        percent_mode : {'percent', None}, optional
            If 'percent', format y-axis labels as percentages (default None).
    
        Returns
        -------
        matplotlib.axes.Axes
            The axes object of the created plot.
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        x_col='group'
        y_col='value'
        df_dict = {
            x_col: np.concatenate([[xticklabels[i]]*len(data[i]) for i in range(len(data))]),
            y_col: np.concatenate(data)}
        df = pd.DataFrame(df_dict)

        
        if plot=='violin':
            v=sns.violinplot(data=df,x=x_col,y=y_col,order=xticklabels,palette=palette,inner=None,width=0.4,ax=self.ax)# violin
            for violin in v.collections:
                violin.set_alpha(0.3)   # Set transparency (alpha) for each PolyCollection
        elif plot=='bar':
            sns.barplot(data=df,x=x_col,y=y_col,order=xticklabels,palette=palette,alpha=0.3,ci=0,ax=self.ax)
        
        # errorbar + mean
        means, sems = [], []
        for i, val in enumerate(xticklabels):
            group = df[df[x_col] == val][y_col]
            mean = group.mean()
            sem = group.std(ddof=1) / np.sqrt(len(group))
            means.append(mean)
            sems.append(sem)
            self.ax.plot(i, mean, 'o', color='black', markersize=6, zorder=3) #mean
            self.ax.errorbar(i, mean, yerr=sem, color='black', capsize=15,
                        fmt='none', zorder=20, capthick=4, elinewidth=4)# error bar
        # scatter (jitter)
        sns.stripplot(data=df,x=x_col,y=y_col,order=xticklabels,jitter=True,color='white',edgecolor='black',linewidth=1.5, 
                      alpha=0.5,size=6,zorder=3,ax=self.ax)
        self.set_plot_style(self.ax,xlabel=xlabel,ylabel=ylabel,xticklabels=xticklabels,
                            fontsize_xlabel=fontsize_xlabel,fontsize_ylabel=fontsize_ylabel,fontsize_xtick=fontsize_xtick,fontsize_ytick=fontsize_ytick,
                            y_major_locator=y_major_locator,x_rotation=x_rotation,percent_mode=percent_mode)
        return self

    def two_factor_with_hue(self, df, x, y, hue,plot='bar', palette=None, figsize=(6,5),xlabel='', ylabel='',
                             xticklabels=None, fontsize_xlabel=30,fontsize_ylabel=30,fontsize_xtick=28,fontsize_ytick=28,
                             y_major_locator=3,x_rotation=0,percent_mode=None,legend_loc=(1,0.9),legend_fontsize=20):
        '''
        Parameters
        ----------
        df : pd.DataFrame
            Long-format DataFrame containing the data to plot.(outcome from data_to_long)
        x : str
            Column name for the x-axis factor.
        y : str
            Column name for the dependent variable to plot.
        hue : str
            Column name for grouping (hue) factor.
        palette : dict or list, optional
            Colors to use for the bars for each hue level. Default is None (matplotlib default palette).
        figsize : tuple, optional
            Figure size in inches (width, height). Default is (6, 5).
        xlabel : str, optional
            Label for the x-axis. Default is ''.
        ylabel : str, optional
            Label for the y-axis. Default is ''.
        xticklabels : list, optional
            Custom labels for x-axis ticks. Default is None (uses categorical names from `x`).
        fontsize_xlabel : int, optional
            Font size for the x-axis label. Default is 30.
        fontsize_ylabel : int, optional
            Font size for the y-axis label. Default is 30.
        fontsize_xtick : int, optional
            Font size for x-axis tick labels. Default is 28.
        fontsize_ytick : int, optional
            Font size for y-axis tick labels. Default is 28.
        y_major_locator : int, optional
            Interval for major ticks on y-axis. Default is 3.
        x_rotation : int, optional
            Rotation angle for x-axis tick labels. Default is 0.
        percent_mode : bool, optional
            If True, y-axis values are formatted as percentages. Default is None.
        legend_loc : tuple or str, optional
            Legend location, either a string (e.g., 'upper right') or coordinates (x, y). Default is (1, 0.9).
        legend_fontsize : int, optional
            Font size for legend labels. Default is 20.
    
        Returns
        -------
        self : object
            Returns the current object instance for method chaining.
    
        Notes
        -----
        - Error bars represent mean ± SEM for each condition.
        - Scatter points are jittered and overlaid to show individual data distribution.
        - Legend is automatically adjusted to avoid duplication when multiple layers are plotted.
        ''' 
        summary = df.groupby([x, hue])[y].agg(['mean', 'sem']).reset_index()
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if plot =='bar':
            sns.barplot(data=df,x=x,y=y,hue=hue,ci=None, palette=palette,capsize=0.1,alpha=0.3,ax=self.ax)
        elif plot=='line':
            n_hue = len(df[hue].unique())
            x_unique = df[x].unique()
            width = 0.4  # Hue offset for separating lines/points within each x-category
            for j, hj in enumerate(df[hue].unique()):
                x_pos = [i + j*width - width*(n_hue-1)/2 for i in range(len(x_unique))]
                y_mean = [summary[(summary[x]==xi) & (summary[hue]==hj)]['mean'].item() for xi in x_unique]
                y_sem  = [summary[(summary[x]==xi) & (summary[hue]==hj)]['sem'].item() for xi in x_unique]
        
                self.ax.plot(x_pos, y_mean, marker='o', label=hj,linewidth=5, 
                             color=palette[hj] if palette else None, alpha=0.8, zorder=5)

        
        sns.stripplot(data=df,x=x,y=y,hue=hue,jitter=True, dodge=True,palette=["white"],
                      marker='o', edgecolor='black', color='white',
                        size=6, linewidth=1, zorder=3,alpha=0.5,ax=self.ax)
        x_coords = []
        for i, cat in enumerate(df[x].unique()):
            for j, h in enumerate(df[hue].unique()):
                x_coords.append((i, j))
    
        for i, xi in enumerate(df[x].unique()):
            for j, hj in enumerate(df[hue].unique()):
                bar_x = i + j*0.4 - 0.4*(len(df[hue].unique())-1)/2 # center x point of the bar

                mean = summary[(summary[x]==xi) & (summary[hue]==hj)]['mean'].item()
                sem  = summary[(summary[x]==xi) & (summary[hue]==hj)]['sem'].item()
                if plot =='bar':
                    self.ax.errorbar(bar_x, mean, yerr=sem, color='black', capsize=15,
                            fmt='none', zorder=20, capthick=4, elinewidth=4)# error bar
                elif plot =='line':
                    self.ax.errorbar(bar_x, mean, yerr=sem, color=palette[hj], capsize=15,
                            fmt='none', zorder=20, capthick=4, elinewidth=4)
    
        handles, labels = self.ax.get_legend_handles_labels()
        if plot =='bar':
            self.ax.legend(handles[-len(df[hue].unique()):], labels[-len(df[hue].unique()):], title=None,frameon=False,loc=legend_loc,fontsize=legend_fontsize, )
        elif plot =='line':
            self.ax.legend(handles[:len(df[hue].unique()):], labels[:len(df[hue].unique()):], title=None,frameon=False,loc=legend_loc,fontsize=legend_fontsize, )
        self.set_plot_style(self.ax,xlabel=xlabel,ylabel=ylabel,xticklabels=xticklabels,
                                fontsize_xlabel=fontsize_xlabel,fontsize_ylabel=fontsize_ylabel,fontsize_xtick=fontsize_xtick,fontsize_ytick=fontsize_ytick,
                                y_major_locator=y_major_locator,x_rotation=x_rotation,percent_mode=percent_mode)
        return self