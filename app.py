import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols, logit
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Clinical Statistical Analysis Tool", layout="wide")

st.title("🩺 Clinical Statistical Analysis Tool")
st.caption("Upload a dataset (CSV or Excel) and run common clinical/biostatistical analyses.")

# ---------- Helper functions ----------

def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)

def is_numeric(series):
    return pd.api.types.is_numeric_dtype(series)

def is_categorical(series, max_unique=20):
    return (not is_numeric(series)) or series.nunique() <= max_unique

# ---------- Sidebar: Upload ----------

st.sidebar.header("1. Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])

if "df" not in st.session_state:
    st.session_state.df = None

if uploaded_file is not None:
    try:
        st.session_state.df = load_data(uploaded_file)
        st.sidebar.success(f"Loaded {uploaded_file.name} ({st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns)")
    except Exception as e:
        st.sidebar.error(f"Error loading file: {e}")

df = st.session_state.df

if df is None:
    st.info("👈 Upload a CSV or Excel file from the sidebar to get started.")
    st.markdown("""
    ### What this tool can do
    - **Descriptive statistics** — mean, SD, median, IQR, missing values, frequency tables
    - **Compare two groups** — independent t-test, Mann-Whitney U test
    - **Compare 3+ groups** — one-way ANOVA, Kruskal-Wallis
    - **Categorical association** — Chi-square test, Fisher's exact test
    - **Correlation** — Pearson / Spearman
    - **Regression** — simple/multiple linear regression, logistic regression
    - **Visualizations** — histograms, boxplots, scatter plots, correlation heatmap

    Your data is processed only in this session and is not stored permanently.
    """)
    st.stop()

# ---------- Data Preview ----------

st.subheader("📋 Data Preview")
st.dataframe(df.head(20), use_container_width=True)

col1, col2, col3 = st.columns(3)
col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing values", int(df.isna().sum().sum()))

numeric_cols = [c for c in df.columns if is_numeric(df[c])]
categorical_cols = [c for c in df.columns if is_categorical(df[c])]

# ---------- Sidebar: Analysis selection ----------

st.sidebar.header("2. Choose Analysis")
analysis = st.sidebar.selectbox(
    "Select an analysis type",
    [
        "Descriptive Statistics",
        "Compare Two Groups (t-test / Mann-Whitney)",
        "Compare 3+ Groups (ANOVA / Kruskal-Wallis)",
        "Categorical Association (Chi-square / Fisher's)",
        "Correlation (Pearson / Spearman)",
        "Linear Regression",
        "Logistic Regression",
    ],
)

st.divider()

# ---------- Descriptive Statistics ----------

if analysis == "Descriptive Statistics":
    st.subheader("📊 Descriptive Statistics")

    if numeric_cols:
        st.markdown("**Numeric variables**")
        desc = df[numeric_cols].describe().T
        desc["missing"] = df[numeric_cols].isna().sum()
        desc["median"] = df[numeric_cols].median()
        desc["IQR"] = df[numeric_cols].quantile(0.75) - df[numeric_cols].quantile(0.25)
        st.dataframe(desc, use_container_width=True)

        col_to_plot = st.selectbox("Visualize a numeric variable", numeric_cols)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        sns.histplot(df[col_to_plot].dropna(), kde=True, ax=axes[0])
        axes[0].set_title(f"Distribution of {col_to_plot}")
        sns.boxplot(y=df[col_to_plot], ax=axes[1])
        axes[1].set_title(f"Boxplot of {col_to_plot}")
        st.pyplot(fig)

    if categorical_cols:
        st.markdown("**Categorical variables**")
        cat_col = st.selectbox("Select a categorical variable", categorical_cols)
        freq = df[cat_col].value_counts(dropna=False).rename("count").to_frame()
        freq["percent"] = (freq["count"] / freq["count"].sum() * 100).round(1)
        st.dataframe(freq, use_container_width=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x=cat_col, data=df, ax=ax, order=freq.index)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# ---------- Compare Two Groups ----------

elif analysis == "Compare Two Groups (t-test / Mann-Whitney)":
    st.subheader("⚖️ Compare Two Groups")
    group_col = st.selectbox("Grouping variable (categorical, must have exactly 2 groups)", categorical_cols)
    outcome_col = st.selectbox("Outcome variable (numeric)", numeric_cols)

    groups = df[group_col].dropna().unique()
    if len(groups) != 2:
        st.warning(f"Selected grouping variable has {len(groups)} categories. Please choose a variable with exactly 2 groups.")
    else:
        g1_name, g2_name = groups[0], groups[1]
        g1 = df[df[group_col] == g1_name][outcome_col].dropna()
        g2 = df[df[group_col] == g2_name][outcome_col].dropna()

        st.markdown(f"Comparing **{outcome_col}** between **{g1_name}** (n={len(g1)}) and **{g2_name}** (n={len(g2)})")

        summary = pd.DataFrame({
            "Group": [g1_name, g2_name],
            "n": [len(g1), len(g2)],
            "Mean": [g1.mean(), g2.mean()],
            "SD": [g1.std(), g2.std()],
            "Median": [g1.median(), g2.median()],
        })
        st.dataframe(summary, use_container_width=True)

        # Normality check
        norm1 = stats.shapiro(g1)[1] if len(g1) >= 3 and len(g1) <= 5000 else None
        norm2 = stats.shapiro(g2)[1] if len(g2) >= 3 and len(g2) <= 5000 else None

        t_stat, t_p = stats.ttest_ind(g1, g2, equal_var=False)
        u_stat, u_p = stats.mannwhitneyu(g1, g2, alternative="two-sided")

        st.markdown("**Independent t-test (Welch's, unequal variance)**")
        st.write(f"t = {t_stat:.3f}, p-value = {t_p:.4f}")

        st.markdown("**Mann-Whitney U test (non-parametric alternative)**")
        st.write(f"U = {u_stat:.3f}, p-value = {u_p:.4f}")

        if norm1 is not None and norm2 is not None:
            st.caption(f"Shapiro-Wilk normality test p-values: {g1_name}={norm1:.4f}, {g2_name}={norm2:.4f}. "
                        f"If either p < 0.05, data may not be normally distributed — consider the Mann-Whitney result.")

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=group_col, y=outcome_col, data=df, ax=ax)
        sns.stripplot(x=group_col, y=outcome_col, data=df, ax=ax, color="black", alpha=0.4, size=3)
        st.pyplot(fig)

# ---------- Compare 3+ Groups ----------

elif analysis == "Compare 3+ Groups (ANOVA / Kruskal-Wallis)":
    st.subheader("📐 Compare 3+ Groups")
    group_col = st.selectbox("Grouping variable (categorical, 3+ groups)", categorical_cols)
    outcome_col = st.selectbox("Outcome variable (numeric)", numeric_cols)

    groups_data = [df[df[group_col] == g][outcome_col].dropna() for g in df[group_col].dropna().unique()]
    group_names = df[group_col].dropna().unique()

    if len(groups_data) < 3:
        st.warning("Selected variable has fewer than 3 groups. Use the two-group comparison instead.")
    else:
        summary = pd.DataFrame({
            "Group": group_names,
            "n": [len(g) for g in groups_data],
            "Mean": [g.mean() for g in groups_data],
            "SD": [g.std() for g in groups_data],
        })
        st.dataframe(summary, use_container_width=True)

        f_stat, f_p = stats.f_oneway(*groups_data)
        h_stat, h_p = stats.kruskal(*groups_data)

        st.markdown("**One-way ANOVA**")
        st.write(f"F = {f_stat:.3f}, p-value = {f_p:.4f}")

        st.markdown("**Kruskal-Wallis test (non-parametric alternative)**")
        st.write(f"H = {h_stat:.3f}, p-value = {h_p:.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        sns.boxplot(x=group_col, y=outcome_col, data=df, ax=ax)
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig)

# ---------- Categorical Association ----------

elif analysis == "Categorical Association (Chi-square / Fisher's)":
    st.subheader("🔗 Categorical Association")
    var1 = st.selectbox("Variable 1", categorical_cols, key="cat1")
    var2 = st.selectbox("Variable 2", categorical_cols, key="cat2", index=min(1, len(categorical_cols)-1))

    if var1 == var2:
        st.warning("Please select two different variables.")
    else:
        contingency = pd.crosstab(df[var1], df[var2])
        st.markdown("**Contingency table**")
        st.dataframe(contingency, use_container_width=True)

        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        st.markdown("**Chi-square test**")
        st.write(f"Chi2 = {chi2:.3f}, df = {dof}, p-value = {p:.4f}")

        if contingency.shape == (2, 2):
            odds_ratio, fisher_p = stats.fisher_exact(contingency)
            st.markdown("**Fisher's Exact Test (2x2 table)**")
            st.write(f"Odds ratio = {odds_ratio:.3f}, p-value = {fisher_p:.4f}")

        fig, ax = plt.subplots(figsize=(7, 4))
        contingency.plot(kind="bar", stacked=True, ax=ax)
        st.pyplot(fig)

# ---------- Correlation ----------

elif analysis == "Correlation (Pearson / Spearman)":
    st.subheader("📈 Correlation Analysis")
    var1 = st.selectbox("Variable 1", numeric_cols, key="corr1")
    var2 = st.selectbox("Variable 2", numeric_cols, key="corr2", index=min(1, len(numeric_cols)-1))

    sub = df[[var1, var2]].dropna()
    pear_r, pear_p = stats.pearsonr(sub[var1], sub[var2])
    spear_r, spear_p = stats.spearmanr(sub[var1], sub[var2])

    st.write(f"**Pearson r** = {pear_r:.3f}, p-value = {pear_p:.4f}")
    st.write(f"**Spearman rho** = {spear_r:.3f}, p-value = {spear_p:.4f}")

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.regplot(x=var1, y=var2, data=sub, ax=ax, scatter_kws={"alpha": 0.5})
    st.pyplot(fig)

    if len(numeric_cols) > 2:
        st.markdown("**Correlation heatmap (all numeric variables)**")
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        sns.heatmap(df[numeric_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax2)
        st.pyplot(fig2)

# ---------- Linear Regression ----------

elif analysis == "Linear Regression":
    st.subheader("📉 Linear Regression")
    outcome = st.selectbox("Dependent variable (numeric outcome)", numeric_cols)
    predictors = st.multiselect("Independent variables (predictors)", [c for c in df.columns if c != outcome])

    if predictors:
        sub = df[[outcome] + predictors].dropna()
        formula = f"{outcome} ~ " + " + ".join([f"C({p})" if df[p].dtype == "object" else p for p in predictors])
        try:
            model = ols(formula, data=sub).fit()
            st.text(model.summary())
        except Exception as e:
            st.error(f"Model error: {e}")
    else:
        st.info("Select at least one predictor variable.")

# ---------- Logistic Regression ----------

elif analysis == "Logistic Regression":
    st.subheader("🎯 Logistic Regression")
    st.caption("Outcome should be binary (e.g., 0/1, Yes/No).")
    outcome = st.selectbox("Dependent variable (binary outcome)", categorical_cols)
    predictors = st.multiselect("Independent variables (predictors)", [c for c in df.columns if c != outcome])

    if predictors:
        sub = df[[outcome] + predictors].dropna()
        unique_vals = sub[outcome].unique()
        if len(unique_vals) != 2:
            st.warning(f"Outcome variable has {len(unique_vals)} categories. Logistic regression requires exactly 2.")
        else:
            sub["_outcome_binary"] = (sub[outcome] == unique_vals[1]).astype(int)
            formula = "_outcome_binary ~ " + " + ".join([f"C({p})" if df[p].dtype == "object" else p for p in predictors])
            try:
                model = logit(formula, data=sub).fit(disp=0)
                st.text(model.summary())
                st.markdown("**Odds Ratios**")
                odds = pd.DataFrame({
                    "OR": np.exp(model.params),
                    "2.5% CI": np.exp(model.conf_int()[0]),
                    "97.5% CI": np.exp(model.conf_int()[1]),
                    "p-value": model.pvalues,
                })
                st.dataframe(odds, use_container_width=True)
            except Exception as e:
                st.error(f"Model error: {e}")
    else:
        st.info("Select at least one predictor variable.")

st.divider()
st.caption("⚠️ This tool is for exploratory and educational use. Always verify statistical assumptions and consult a biostatistician for publication-grade analysis.")
