from pathlib import Path
import warnings, math
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[1]
DATA, FIG, TAB = ROOT / 'data', ROOT / 'figures', ROOT / 'tables'
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)


def load_data():
    return pd.read_csv(DATA/'cleaned_data.csv'), pd.read_csv(DATA/'cleaned_replication_data.csv')


def numericize(df):
    cols = ['pro_party','anti_party','neutral_news','message_greater','message_less','net_party','partisan','moderate','fake_news','true_news','pro_performance','anti_performance','polarizing']
    for c in cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)
    if 'prob_true' in df: df['prob_true'] = pd.to_numeric(df['prob_true'], errors='coerce')
    return df


def add_demeaned(df):
    if 'prob_true_demeaned' not in df.columns:
        df['prob_true_mean'] = df.groupby('code')['prob_true'].transform('mean')
        df['prob_true_demeaned'] = df['prob_true'] - df['prob_true_mean']
    if 'abs_net_party' not in df.columns and 'net_party' in df: df['abs_net_party'] = df['net_party'].abs()
    return df


def add_follow(df):
    need = {'your_reguess','your_answer','message_greater','message_less'}
    if 'follow_message' in df.columns or not need.issubset(df.columns): return df
    up = ((df.message_greater==1)&(df.your_reguess>df.your_answer))|((df.message_less==1)&(df.your_reguess<df.your_answer))
    same = df.your_reguess == df.your_answer
    down = ((df.message_greater==1)&(df.your_reguess<df.your_answer))|((df.message_less==1)&(df.your_reguess>df.your_answer))
    df['follow_message'] = np.select([up, same, down], [1, 0, -1], default=np.nan)
    return df


def add_vars(df):
    df = df.copy()
    return add_follow(add_demeaned(numericize(df)))


def demean_once(arr, g):
    codes = pd.Categorical(g).codes; valid = codes >= 0
    for j in range(arr.shape[1]):
        sums = np.bincount(codes[valid], weights=arr[valid, j]); counts = np.bincount(codes[valid])
        means = np.divide(sums, counts, out=np.zeros_like(sums, dtype=float), where=counts != 0)
        arr[valid, j] -= means[codes[valid]]


def residualize_matrix(mat, groups, max_iter=100, tol=1e-10):
    arr = np.asarray(mat, dtype=float).copy(); prev = arr.copy()
    for _ in range(max_iter):
        for g in groups: demean_once(arr, g)
        if np.nanmax(np.abs(arr-prev)) < tol: break
        prev = arr.copy()
    return arr


class SimpleResult:
    def __init__(self, params, bse, pvalues, rsquared):
        self.params, self.bse, self.pvalues, self.rsquared = params, bse, pvalues, rsquared


def norm_sf_abs(x):
    return 0.5 * math.erfc(abs(float(x)) / math.sqrt(2))


def prep_hdfe(y, Xcols, data, fe_cols, cluster_col):
    cols = list(dict.fromkeys([y] + Xcols + list(fe_cols) + [cluster_col]))
    d = data[cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) == 0: raise ValueError('No observations after dropping missing values')
    R = residualize_matrix(d[[y]+Xcols].astype(float).values, [d[c] for c in fe_cols])
    return d, R[:,0], R[:,1:]


def ols_core(yr, Xr, Xcols):
    keep = np.where(Xr.std(axis=0) > 1e-12)[0]; X = Xr[:, keep]
    beta = np.linalg.lstsq(X, yr, rcond=None)[0]; resid = yr - X @ beta
    return X, beta, resid, [Xcols[i] for i in keep]


def cluster_vcov(X, resid, clusters):
    n, k = X.shape; cl = pd.Categorical(clusters).codes; meat = np.zeros((k,k))
    for c in np.unique(cl):
        S = X[cl==c].T @ resid[cl==c]; meat += np.outer(S, S)
    G = len(np.unique(cl)); scale = (G/(G-1))*((n-1)/(n-k)) if G > 1 and n > k else 1.0
    return scale * np.linalg.pinv(X.T @ X) @ meat @ np.linalg.pinv(X.T @ X)


def result_from(beta, se, names, yr, resid):
    pvals = [2*norm_sf_abs(b/s) if s > 0 else np.nan for b, s in zip(beta, se)]
    rsq = 1 - np.sum(resid**2)/np.sum((yr-yr.mean())**2) if np.sum((yr-yr.mean())**2) > 0 else np.nan
    return SimpleResult(pd.Series(beta,index=names), pd.Series(se,index=names), pd.Series(pvals,index=names), rsq)


def hdfe_ols(y, Xcols, data, fe_cols=('code','topic_id','round_number'), cluster_col='code'):
    d, yr, Xr = prep_hdfe(y, Xcols, data, fe_cols, cluster_col)
    X, beta, resid, names = ols_core(yr, Xr, Xcols)
    se = np.sqrt(np.diag(cluster_vcov(X, resid, d[cluster_col])))
    return result_from(beta, se, names, yr, resid), len(d)


def parse_formula(formula, data):
    lhs, rhs = [s.strip() for s in formula.split('~')]; xcols, fe_cols, d = [], [], data.copy()
    for t in [v.strip() for v in rhs.split('+')]:
        xcols, fe_cols, d = parse_term(t, xcols, fe_cols, d)
    return lhs, xcols, fe_cols, d


def parse_term(t, xcols, fe_cols, d):
    if t.startswith('C('): fe_cols.append(t[2:].strip('() '))
    elif ':' in t:
        a, b = [x.strip() for x in t.split(':')]; d[f'{a}:{b}'] = d[a].astype(float)*d[b].astype(float); xcols.append(f'{a}:{b}')
    elif t and t != '1': xcols.append(t)
    return xcols, fe_cols, d


def cluster_reg(formula, data):
    lhs = formula.split('~')[0].strip()
    d = data.replace([np.inf, -np.inf], np.nan).dropna(subset=[lhs]).copy()
    res = smf.ols(formula, data=d).fit(cov_type='cluster', cov_kwds={'groups': d['code']})
    return res, int(res.nobs)


def coef(res, var):
    return res.params.get(var, np.nan), res.bse.get(var, np.nan), res.pvalues.get(var, np.nan)

def stars(p):
    return '***' if p < 0.01 else ('**' if p < 0.05 else ('*' if p < 0.1 else ''))


def stat_value(label, col, fun, base, df):
    source = base if col in base.columns and label not in ['News assessments','Mean Pr(True News)','SD Pr(True News)'] else df
    if col not in source.columns: return np.nan
    if fun in ['nunique','count','mean','std']: return getattr(source[col], fun)()
    return fun(source[col])


def summary_specs():
    return [('Subjects','code','nunique'),('News assessments','prob_true','count'),('Mean Pr(True News)','prob_true','mean'),('SD Pr(True News)','prob_true','std'),('Age','age','mean'),('Male','male','mean'),('White','white','mean'),('College years / education','edu','mean'),('Log income','log_inc','mean'),('Pro-Dem share','pro_dem','mean'),('Pro-Rep share','pro_rep','mean'),('Neutral party share','net_party',lambda s:(s==0).mean())]


def make_summary_stats(df):
    base = df.drop_duplicates('code')
    rows = [{'Statistic': lab, 'Value': stat_value(lab, col, fun, base, df)} for lab, col, fun in summary_specs()]
    out = pd.DataFrame(rows); out.to_csv(TAB/'summary_stats.csv', index=False)
    return out


def savefig(name):
    plt.tight_layout(); plt.savefig(FIG/name, dpi=200); plt.close()


def cdf_series(s):
    x = np.sort(s.dropna()); y = np.arange(1, len(x)+1) / len(x)
    return x, y


def fig1(df):
    d = df[(df.net_party!=0)&((df.pro_party+df.anti_party)>0)&df.prob_true.notna()]
    plt.figure(figsize=(6,4))
    for lab, sub in [('Pro-Party news', d[d.pro_party==1]), ('Anti-Party news', d[d.anti_party==1])]:
        x, y = cdf_series(sub.prob_true); plt.step(x, y, where='post', label=lab)
    plt.xlabel('Belief about Pr(True News)'); plt.ylabel('Share of responses'); plt.legend(); savefig('figure1_cdf_pro_antiparty.png')


def bar_values(d, groups, ycol):
    vals = [(lab+f'\nn={len(s):,}', s.mean(), 1.96*s.std()/np.sqrt(len(s))) for lab, mask in groups for s in [d.loc[mask, ycol].dropna()]]
    return list(zip(*vals))


def bar_plot(labels, means, ses, name, ylim=(-.1,.1)):
    plt.figure(figsize=(9,4.5)); plt.bar(range(len(labels)), means, yerr=ses, capsize=3)
    plt.axhline(0, lw=.8); plt.xticks(range(len(labels)), labels, fontsize=8)
    plt.ylabel('Pr(True), demeaned'); plt.ylim(*ylim); savefig(name)


def fig2_groups(d):
    return [('Anti-Party,\nPartisans',(d.anti_party==1)&(d.partisan==1)),('Anti-Party,\nModerates',(d.anti_party==1)&(d.moderate==1)),('Neutral,\nAll Subjects',d.neutral_news==1),('Pro-Party,\nModerates',(d.pro_party==1)&(d.moderate==1)),('Pro-Party,\nPartisans',(d.pro_party==1)&(d.partisan==1))]


def fig2(df):
    d = df[(df.net_party!=0)&((df.pro_party+df.anti_party+df.neutral_news)>0)&((df.message_greater+df.message_less)>0)].copy()
    labels, means, ses = bar_values(d, fig2_groups(d), 'prob_true_demeaned')
    bar_plot(labels, means, ses, 'figure2_motivated_reasoning_partisanship.png', (-.1,.1))


def fig3_groups(d):
    return [('Anti-Party,\nFake News',(d.anti_party==1)&(d.fake_news==1)),('Anti-Party,\nTrue News',(d.anti_party==1)&(d.true_news==1)),('Neutral,\nFake News',(d.neutral_news==1)&(d.fake_news==1)),('Neutral,\nTrue News',(d.neutral_news==1)&(d.true_news==1)),('Pro-Party,\nFake News',(d.pro_party==1)&(d.fake_news==1)),('Pro-Party,\nTrue News',(d.pro_party==1)&(d.true_news==1))]


def fig3(df):
    d = df[(df.net_party!=0)&(df.ego_news!=1)&((df.message_greater+df.message_less)>0)].copy()
    labels, means, ses = bar_values(d, fig3_groups(d), 'prob_true_demeaned')
    bar_plot(labels, means, ses, 'figure3_fake_true_news.png', (-.1,.1))


def topic_terms():
    tv = ['good_topic13', 'good_topic1415', 'good_topic8', 'good_topic4', 'good_topic6', 'good_topic3', 'good_topic2', 'good_topic5', 'good_topic7', 'good_topic1']
    tl = ['Own performance', 'Party score', 'Media', 'Gun laws', 'Gender', 'Obama crime', 'Refugees', 'Race', 'Mobility', 'Climate']
    return tv, tl


def topic_estimates(df):
    tv, tl = topic_terms(); use = [v for v in tv if v in df.columns]
    data = df[df.net_party!=0].dropna(subset=['prob_true']).copy()
    res, _ = hdfe_ols('prob_true', use, data, fe_cols=('round_number','topic_id','code'))
    labs = [tl[tv.index(v)] for v in use]
    return use, labs, [res.params[v] for v in use], [res.bse[v] for v in use], [res.pvalues[v] for v in use]


def fig4(df):
    try: use, labs, coefs, ses, p = topic_estimates(df)
    except Exception as e: print('Figure 4 regression skipped:', e); return
    plt.figure(figsize=(7,5)); y = np.arange(len(use)); plt.errorbar(coefs, y, xerr=1.96*np.array(ses), fmt='o')
    plt.axvline(0, lw=.8); plt.yticks(y, labs); plt.xlabel('Effect of Pro-Party/Performance news on Pr(True)'); savefig('figure4_by_topic.png')
    pd.DataFrame({'term':use,'label':labs,'coef':coefs,'se':ses,'p':p}).to_csv(TAB/'figure4_topic_coefficients.csv', index=False)


def make_figures(df):
    for f in [fig1, fig2, fig3, fig4]: f(df)


def main_data(df):
    return df[(df.net_party!=0)&((df.pro_party+df.anti_party)==1)&df.prob_true.notna()].copy()


def neutral_data(df):
    return df[(df.net_party!=0)&((df.pro_party+df.anti_party+df.neutral_news)>=1)&df.prob_true.notna()].copy()

def fe_full():
    return ' + C(code) + C(topic_id) + C(round_number)'


def fe_no_subject():
    return ' + C(topic_id) + C(round_number)'


def fe_no_topic():
    return ' + C(code) + C(round_number)'


def controls():
    return ['white','black','asian','latino','age','male','edu','red_state','religious_group','log_inc']

def controls_rhs(d):
    cs = ['white', 'black', 'asian', 'latino', 'age', 'male',
          'education', 'log_income', 'religious', 'trump_state']
    cs = [c for c in cs if c in d.columns]
    return '' if not cs else ' + ' + ' + '.join(cs)

def table2_specs(d1, d4):
    c = controls_rhs(d1)
    return [
        ('Table 2 col.1','prob_true ~ pro_party' + c + ' + C(topic_id) + C(round_number)',d1,['pro_party']),
        ('Table 2 col.2','prob_true ~ pro_party + C(code) + C(topic_id) + C(round_number)',d1,['pro_party']),
        ('Table 2 col.3','prob_true ~ pro_party + abs_net_party:pro_party + C(code) + C(topic_id) + C(round_number)',d1,['pro_party','abs_net_party:pro_party']),
        ('Table 2 col.4','prob_true ~ pro_party + anti_party + C(code) + C(round_number)',d4,['pro_party','anti_party']),
        ('Table 2 col.5','prob_true ~ true_news + C(code) + C(topic_id) + C(round_number)',d1,['true_news']),
        ('Table 2 col.6','prob_true ~ pro_party + true_news + C(code) + C(topic_id) + C(round_number)',d1,['pro_party','true_news'])
    ]
def add_abs_fe_interactions(d):
    d = d.copy()
    for c in ['round_number', 'topic_id']:
        x = pd.get_dummies(d[c], prefix=f'abs_{c}', dtype=float)
        d = pd.concat([d, x.mul(d.abs_net_party, axis=0)], axis=1)
    return d


def abs_fe_cols(d):
    return [c for c in d.columns if c.startswith('abs_round_number_') or c.startswith('abs_topic_id_')]

def run_spec(name, formula, data, vars_, ycol):
    try: res, n = cluster_reg(formula, data)
    except Exception as e: return [{'spec':name,'term':'ERROR','coef':np.nan,'se':np.nan,'p_value':np.nan,'stars':str(e),'n':len(data),'r2':np.nan,'mean_y':np.nan}]
    return [row_from_result(name, res, n, data, v, ycol) for v in vars_]


def row_from_result(name, res, n, data, v, ycol):
    b, se, p = coef(res, v)
    return {'spec':name,'term':v,'coef':b,'se':se,'p_value':p,'stars':stars(p),'n':n,'r2':res.rsquared,'mean_y':data[ycol].mean()}


def write_table2(df):
    rows = [r for spec in table2_specs(main_data(df), neutral_data(df)) for r in run_spec(*spec, ycol='prob_true')]
    out = pd.DataFrame(rows); out.to_csv(TAB/'main_regressions_table2_python.csv', index=False)
    return out


def guess_y(d):
    return 'change_guess_message' if 'change_guess_message' in d.columns else 'follow_message'


def second_guess_data(df):
    y = guess_y(df)
    return df[(df.net_party!=0)&((df.pro_party+df.anti_party)==1)&df[y].notna()].copy()

def table3_rows():
    return [('col.1','pro_party',['pro_party']), ('col.2','polarizing',['polarizing']),
            ('col.3','pro_party + polarizing',['pro_party','polarizing']),
            ('col.4','pro_party + prob_true',['pro_party','prob_true']),
            ('col.5','polarizing + prob_true',['polarizing','prob_true']),
            ('col.6','pro_party + polarizing + prob_true',['pro_party','polarizing','prob_true'])]


def table3_specs(d):
    y = guess_y(d)
    return [(f'Table 3 {n}', y + ' ~ ' + rhs + fe_full(), d, keep)
            for n, rhs, keep in table3_rows()]

def write_table3(df):
    d = second_guess_data(df)
    if len(d) == 0: return
    rows = [r for spec in table3_specs(d) for r in run_spec(*spec, ycol=guess_y(d))]
    pd.DataFrame(rows).to_csv(TAB/'second_guess_table3_python.csv', index=False)

def make_regression_tables(df):
    out = write_table2(df); write_table3(df)
    return out


def replication_check(rep):
    d = main_data(rep); res, n = cluster_reg('prob_true ~ pro_party + C(code) + C(topic_id) + C(round_number)', d)
    row = row_from_result('replication experiment', res, n, d, 'pro_party', 'prob_true')
    row['sample'], row['term'] = 'replication experiment', 'pro_party'
    return row


def write_replication(rep):
    try: rows = [replication_check(rep)]
    except Exception as e: rows = [{'sample':'replication experiment','term':'ERROR','coef':np.nan,'se':np.nan,'p_value':np.nan,'stars':str(e),'n':0,'r2':np.nan,'mean_y':np.nan}]
    pd.DataFrame(rows).to_csv(TAB/'replication_experiment_check.csv', index=False)


def main():
    df, rep = load_data(); df, rep = add_vars(df), add_vars(rep)
    summ = make_summary_stats(df); make_figures(df)
    mainregs = make_regression_tables(df); write_replication(rep)
    print('Wrote outputs to', ROOT); print(summ.to_string(index=False)); print(mainregs.to_string(index=False))


if __name__ == '__main__':
    main()
