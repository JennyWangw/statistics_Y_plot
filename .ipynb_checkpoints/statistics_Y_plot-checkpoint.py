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
        self.alpha = alpha

    # -------------------- 共用工具函数 --------------------
    @staticmethod
    def p_to_sig(p):
        """p 值显著性标注"""
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'
    
    @staticmethod
    def data_to_long(data, factorA_name='FactorA', factorB_name='FactorB', subject_prefix='S'):
        """
        当前函数仅支持2层嵌套/2变量
        
        将嵌套 list 或 array 转成长格式 DataFrame
        外层 list -> factorB 水平
        内层 list -> factorA 水平
        """
        long_data = []
        for b_idx, b_level in enumerate(data, start=1):
            for a_idx, a_list in enumerate(b_level, start=1):
                try:
                    iter(a_list)
                except TypeError:
                    a_list = [a_list]
                for sub_idx, value in enumerate(a_list, start=1):
                    long_data.append({
                        'Subject': f'{subject_prefix}{sub_idx}',
                        factorA_name: f'A{a_idx}',
                        factorB_name: f'B{b_idx}',
                        'Value': value
                    })
        return pd.DataFrame(long_data)

    def run_pairedT_for_posthoc(self, df_long, factor, ttest_func):
        """
        对某个 factor/变量内多水平 做配对 t 检验（重复测量）
        p-Bonferroni 校正
        """
        levels = df_long[factor].unique() #factor是一个变量，levels是该变量的多水平
        results = []
        for a, b in combinations(levels, 2):
            vals_a = df_long[df_long[factor] == a].sort_values('Subject')['Value'].values
            vals_b = df_long[df_long[factor] == b].sort_values('Subject')['Value'].values
            # 删除 NaN
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


    def ttest_1samp_with_precheck(self, data, popmean=0, nan_policy='omit', return_summary=True):
        """
        - 预处理删除有nan的值，并报告删除几个
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
                'mean': f"{np.mean(data_clean):.4f}",  # 保留四位小数
                'std': f"{np.std(data_clean, ddof=1):.4f}", 
                'N_clean': len(data_clean) #删除nan后的样本总数
            })
        
        return result


            

    def ttest_ind_with_precheck(self, x, y):
        '''
        - 删除两组数中的nan，并报告删除数
        - 先检验是否方差齐性
            - 若不齐性自动换为Welch，
            - 若齐性采用常规独立t test
        - 自动输出对应检验方法的df自由度，并报告levene方差齐性检验的统计值
        '''
        x, y = np.array(x), np.array(y)
        nan_removed_x = np.isnan(x).sum()
        nan_removed_y = np.isnan(y).sum()
        x, y = x[~np.isnan(x)], y[~np.isnan(y)] #删除两组数中的nan
        if nan_removed_x > 0 or nan_removed_y > 0:# 报告删除多个数（仅在有 NaN 被删除时）
            print(f"Removed {nan_removed_x} NaN(s) from x, {nan_removed_y} NaN(s) from y")

        lev_stat, lev_p = stats.levene(x, y)#方差齐性检验
        n1, n2 = len(x), len(y)
        df1_lev = 2 - 1       # k-1
        df2_lev = n1 + n2 - 2 # N-k
        equal_var = lev_p > self.alpha
        res = stats.ttest_ind(x, y, equal_var=equal_var) #独立t检验/Welch检验
        s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
        if equal_var:
            df_t = n1 + n2 - 2
        else:
            df_t = (s1/n1 + s2/n2)**2 / ((s1/n1)**2/(n1-1) + (s2/n2)**2/(n2-1))
        return {
            't': res.statistic,'p': res.pvalue,'df_t': df_t,#自由度（适用t/Welch）
            'equal_var': equal_var,#True:方差齐性
            'levene_stat': lev_stat, 'levene_p': lev_p,'levene_df1': df1_lev,'levene_df2': df2_lev#Levene 检验统计量和 p 值
        }


    def ttest_rel_with_precheck(self, x, y):
        """
        - 自动删除含 NaN 的被试，并报告删除被试数量，
        - 返回配对t检验 t 值、p 值、自由度
        """
        x, y = np.array(x), np.array(y)
        mask = ~np.isnan(x) & ~np.isnan(y)
        n_removed = len(x) - np.sum(mask)
        x_clean, y_clean = x[mask], y[mask]# 删除含 NaN 的被试
        if n_removed > 0:
            print(f"Removed {n_removed} sample(s) due to NaN in x or y")
        res = stats.ttest_rel(x_clean, y_clean)
        df = len(x_clean) - 1# 自由度 = 样本数 - 1
        return {
            't': res.statistic,
            'p': res.pvalue,
            'df': df,
            'n_removed': n_removed
        }

    
    
    def one_factor_rm_anova(self, *conditions, factor_name='Condition', subject_prefix='S'):
        """
        1.预处理确保所有的对应的sub都有值。若有nan删除该sub，并报告删除了几个sub 
        2.单因素多水平重复测量 ANOVA
        3.若p显著，事后两两配对t检验，Bonferroni校正p
        
        参数:
            *conditions: 各条件的被试数据 (每个为list或array) eg. one_factor_rm_anova(x1,x2,x3)
            factor_name: 因素名称
            subject_prefix: 被试 ID 前缀
        返回:
            (anova_table, posthoc_df)
        """
        data = np.column_stack(conditions)
        n_before = data.shape[0]
        
        mask = ~np.isnan(data).any(axis=1)# 删除任何含有 NaN 的被试
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
        '''
        data:[[a1,a2,a3],[b1,b2,b3],[c1,c2,v3]...] 仅支持2层嵌套/2变量（每个变量可以是多水平）
            data.shape[0]表示第1个变量（factorB）的多水平，
            data[0,:].shape[0]表示第2个变量（factorA）的多水平，
            a1是一个list或者一维array
        - 预处理：删除有条件缺失值的被试，报告删除几个sub
        - 2 way repeated ANOVA
        - 若主效应显著，事后简单效应配对t检验
        
        '''
        df_long = self.data_to_long(data, factorA_name, factorB_name, subject_prefix) #转换数据

        subjects_total = df_long['Subject'].unique()
        factorA_levels = df_long[factorA_name].unique()
        factorB_levels = df_long[factorB_name].unique()
        valid_subjects = []
        for sub in subjects_total:
            df_sub = df_long[df_long['Subject'] == sub]
            if not df_sub['Value'].isna().any() and \
               len(df_sub) == len(factorA_levels) * len(factorB_levels):  # 条件 1: 没有 NaN； 条件 2: 每个 factorA×factorB 组合都有值
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
        for factor in [factorA_name, factorB_name]:
            if aov_table.loc[factor, 'Pr > F'] < self.alpha:
                print(f'\nPost-hoc for {factor}')
                posthoc_dict[factor] = self.run_pairedT_for_posthoc(df_long_clean, factor, self.ttest_rel_with_precheck)
                print(posthoc_dict[factor])
            else:
                posthoc_dict[factor] = None

        return aov_table, posthoc_dict



    #———————————————————————画图—————————————————————————————————————————————
    def set_plot_style(self, ax, xlabel='', ylabel='', xticklabels=None,
                       xlabelpad=15,fontsize_xlabel=30, fontsize_ylabel=30, fontsize_xtick=28,fontsize_ytick=28,
                       spine_width=5, tick_length=15, y_major_locator=3,x_rotation=0,percent_mode=None):
        """
        统一调整 matplotlib 图形样式
        
        ax: matplotlib Axes 对象
        xlabel, ylabel: 坐标轴标签
        xticklabels: 自定义 x 刻度标签
        xlabelpad: x轴标签间距
        fontsize_label: 坐标轴标签字体大小
        fontsize_tick: 刻度字体大小
        spine_width: 坐标轴粗细
        tick_length: 刻度长度
        y_major_locator: y轴主刻度间距
        """
        ax.set_xlabel(xlabel, fontsize=fontsize_xlabel, fontdict={'family': 'Arial'}, labelpad=xlabelpad)
        ax.set_ylabel(ylabel, fontsize=fontsize_ylabel, fontdict={'family': 'Arial'})
        
        if xticklabels is not None:
            plt.xticks(range(len(xticklabels)), labels=xticklabels,rotation=x_rotation)
        plt.xticks(fontsize=fontsize_xtick)
        plt.yticks(fontsize=fontsize_ytick)

        # 坐标轴边框
        ax.spines['left'].set_linewidth(spine_width)
        ax.spines['bottom'].set_linewidth(spine_width)

        # 坐标轴刻度
        plt.tick_params(axis='both', which='major', width=spine_width, length=tick_length)

        # y轴刻度间距
        ax.yaxis.set_major_locator(MultipleLocator(y_major_locator))

        # 设置 y 轴标签为百分位数格式
        def percent_formatter(x, pos):
            return f'{x*100:.0f}%'  # 将数值转换为百分比
        if percent_mode:
            ax.yaxis.set_major_formatter(FuncFormatter(percent_formatter))


        sns.despine()
        plt.tight_layout()


    def plot_with_scatter(
        self, data, palette=None, p_value=None,plot='bar',
        xlabel='', ylabel='', xticklabels=None, figsize=(6,5),
        fontsize_xlabel=30, fontsize_ylabel=30, fontsize_xtick=28,fontsize_ytick=28,y_major_locator=3,x_rotation=0,percent_mode=None,):
        """
        封装 violin/bar + scatter + errorbar 
        
        参数：
        palette: dict, x_col取值对应颜色 eg.{'A':'red','B':'green','C':'blue'},
        p_value: float, 用于显著性标注
        xlabel, ylabel: str，坐标轴标签
        xticklabels: list，自定义 x 刻度标签
        figsize: tuple，画布大小
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
                violin.set_alpha(0.3)   # 对每个 PolyCollection 设置透明度
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

